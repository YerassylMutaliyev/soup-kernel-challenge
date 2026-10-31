"""Rotary embedding fused with the LoRA merge, the fp32<->fp16 casts and the GQA head expansion.

Base Soup, per attention layer, after the q/k/v projections (bitsandbytes returns fp16 cast back
to the fp32 input dtype; peft adds the fp16 LoRA output to the fp32 result):
    q = fp32(q16) + fp32(lora_q16)          v = fp32(v16) + fp32(lora_v16)       k = fp32(k16)
    q, k = q*cos + rotate_half(q)*sin        (fp32, rotate_half = cat(-x2, x1))
    k, v = repeat_kv(k), repeat_kv(v)        (2 KV heads -> 12, fp32 copies)
    q, k, v -> fp16                           (autocast, in SDPA)
That is ~20 element-wise launches and five fp32 copies of q/k/v per layer, all kept for backward.

Here one program per token reads the fp16 q|k|v row of the fused projection GEMM plus the LoRA
outputs and writes the three fp16 SDPA inputs directly, in the same fp32 arithmetic. The KV head
expansion is done by the store (each query head reads its KV head), so no fp32 copy exists.
Backward applies the transposed rotation, sums the expanded heads in fp32 and writes one fp16
[tokens, q|k|v] gradient for the fused GEMM backward (and copies of the q and v parts for the
LoRA branches).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_qkv_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, pos_ptr, cos_ptr, sin_ptr,
                        q_ptr, k_ptr, v_ptr, stride_qkv,
                        H: tl.constexpr, KVH: tl.constexpr, KO: tl.constexpr, DH: tl.constexpr,
                        BH: tl.constexpr, BKO: tl.constexpr,
                        LORA_Q: tl.constexpr, LORA_V: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    p = tl.load(pos_ptr + t).to(tl.int64)
    d = tl.arange(0, DH)
    half: tl.constexpr = DH // 2
    d_partner = (d + half) % DH
    sign = tl.where(d < half, -1.0, 1.0)[None, :]
    c = tl.load(cos_ptr + p * DH + d)[None, :]
    s = tl.load(sin_ptr + p * DH + d)[None, :]
    row = qkv_ptr + t * stride_qkv

    h = tl.arange(0, BH)[:, None]
    hm = h < H
    qa = tl.load(row + h * DH + d[None, :], mask=hm, other=0.0).to(tl.float32)
    qb = tl.load(row + h * DH + d_partner[None, :], mask=hm, other=0.0).to(tl.float32)
    if LORA_Q:
        lrow = lq_ptr + t * (H * DH)
        qa = qa + tl.load(lrow + h * DH + d[None, :], mask=hm, other=0.0).to(tl.float32)
        qb = qb + tl.load(lrow + h * DH + d_partner[None, :], mask=hm, other=0.0).to(tl.float32)
    q_rot = qa * c + (sign * qb) * s
    tl.store(q_ptr + t * (H * DH) + h * DH + d[None, :], q_rot.to(tl.float16), mask=hm)

    j = tl.arange(0, BKO)[:, None]
    jm = j < KO
    g = j // (KO // KVH)  # KV head read by output head j
    ka = tl.load(row + (H + g) * DH + d[None, :], mask=jm, other=0.0).to(tl.float32)
    kb = tl.load(row + (H + g) * DH + d_partner[None, :], mask=jm, other=0.0).to(tl.float32)
    k_rot = ka * c + (sign * kb) * s
    tl.store(k_ptr + t * (KO * DH) + j * DH + d[None, :], k_rot.to(tl.float16), mask=jm)

    va = tl.load(row + (H + KVH + g) * DH + d[None, :], mask=jm, other=0.0).to(tl.float32)
    if LORA_V:
        va = va + tl.load(lv_ptr + t * (KVH * DH) + g * DH + d[None, :], mask=jm,
                          other=0.0).to(tl.float32)
    tl.store(v_ptr + t * (KO * DH) + j * DH + d[None, :], va.to(tl.float16), mask=jm)


@triton.jit
def rope_qkv_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, pos_ptr, sin_ptr, cos_ptr,
                        dqkv_ptr, dlq_ptr, dlv_ptr, stride_dqkv,
                        H: tl.constexpr, KVH: tl.constexpr, KO: tl.constexpr, DH: tl.constexpr,
                        BH: tl.constexpr, BKO: tl.constexpr,
                        LORA_Q: tl.constexpr, LORA_V: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    p = tl.load(pos_ptr + t).to(tl.int64)
    d = tl.arange(0, DH)
    half: tl.constexpr = DH // 2
    d_partner = (d + half) % DH
    neg_sign = tl.where(d < half, 1.0, -1.0)[None, :]
    c = tl.load(cos_ptr + p * DH + d)[None, :]
    s_partner = tl.load(sin_ptr + p * DH + d_partner)[None, :]
    out = dqkv_ptr + t * stride_dqkv

    h = tl.arange(0, BH)[:, None]
    hm = h < H
    ga = tl.load(dq_ptr + t * (H * DH) + h * DH + d[None, :], mask=hm, other=0.0).to(tl.float32)
    gb = tl.load(dq_ptr + t * (H * DH) + h * DH + d_partner[None, :], mask=hm,
                 other=0.0).to(tl.float32)
    dq = (ga * c + neg_sign * (gb * s_partner)).to(tl.float16)
    tl.store(out + h * DH + d[None, :], dq, mask=hm)
    if LORA_Q:
        tl.store(dlq_ptr + t * (H * DH) + h * DH + d[None, :], dq, mask=hm)

    j = tl.arange(0, BKO)[:, None]
    jm = j < KO
    gj = j // (KO // KVH)
    ka = tl.load(dk_ptr + t * (KO * DH) + j * DH + d[None, :], mask=jm, other=0.0).to(tl.float32)
    kb = tl.load(dk_ptr + t * (KO * DH) + j * DH + d_partner[None, :], mask=jm,
                 other=0.0).to(tl.float32)
    va = tl.load(dv_ptr + t * (KO * DH) + j * DH + d[None, :], mask=jm, other=0.0).to(tl.float32)
    for g in tl.static_range(KVH):
        sel = (gj == g) & jm
        ka_g = tl.sum(tl.where(sel, ka, 0.0), axis=0)[None, :]
        kb_g = tl.sum(tl.where(sel, kb, 0.0), axis=0)[None, :]
        v_g = tl.sum(tl.where(sel, va, 0.0), axis=0)[None, :]
        dk = (ka_g * c + neg_sign * (kb_g * s_partner)).to(tl.float16)
        tl.store(out + (H + g) * DH + d[None, :], dk)
        dv = v_g.to(tl.float16)
        tl.store(out + (H + KVH + g) * DH + d[None, :], dv)
        if LORA_V:
            tl.store(dlv_ptr + t * (KVH * DH) + g * DH + d[None, :], dv)


class FusedRopeQKV(torch.autograd.Function):
    """(qkv16 [N, (H+2KVH)*DH], lora_q16, lora_v16) -> q [N,H,DH], k [N,KO,DH], v [N,KO,DH]."""

    @staticmethod
    def forward(ctx, qkv, lora_q, lora_v, pos, cos, sin, n_heads, n_kv, n_out_kv, head_dim):
        n = qkv.shape[0]
        dev = qkv.device
        q = torch.empty(n, n_heads, head_dim, dtype=torch.float16, device=dev)
        k = torch.empty(n, n_out_kv, head_dim, dtype=torch.float16, device=dev)
        v = torch.empty(n, n_out_kv, head_dim, dtype=torch.float16, device=dev)
        lq = lora_q if lora_q is not None else qkv
        lv = lora_v if lora_v is not None else qkv
        if n:
            rope_qkv_fwd_kernel[(n,)](
                qkv, lq, lv, pos, cos, sin, q, k, v, qkv.stride(0), H=n_heads, KVH=n_kv,
                KO=n_out_kv, DH=head_dim, BH=triton.next_power_of_2(n_heads),
                BKO=triton.next_power_of_2(n_out_kv), LORA_Q=lora_q is not None,
                LORA_V=lora_v is not None, num_warps=4)
        ctx.save_for_backward(pos, cos, sin)
        ctx.meta = (n_heads, n_kv, n_out_kv, head_dim, lora_q is not None, lora_v is not None,
                    qkv.shape[1])
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        pos, cos, sin = ctx.saved_tensors
        n_heads, n_kv, n_out_kv, head_dim, has_lq, has_lv, width = ctx.meta
        n = pos.shape[0]
        dev = pos.device
        dq = torch.zeros(n, n_heads, head_dim, dtype=torch.float16, device=dev) if dq is None \
            else dq.contiguous()
        dk = torch.zeros(n, n_out_kv, head_dim, dtype=torch.float16, device=dev) if dk is None \
            else dk.contiguous()
        dv = torch.zeros(n, n_out_kv, head_dim, dtype=torch.float16, device=dev) if dv is None \
            else dv.contiguous()
        dqkv = torch.empty(n, width, dtype=torch.float16, device=dev)
        dlq = torch.empty(n, n_heads * head_dim, dtype=torch.float16, device=dev) if has_lq else None
        dlv = torch.empty(n, n_kv * head_dim, dtype=torch.float16, device=dev) if has_lv else None
        if n:
            rope_qkv_bwd_kernel[(n,)](
                dq, dk, dv, pos, sin, cos, dqkv, dlq if has_lq else dqkv,
                dlv if has_lv else dqkv, dqkv.stride(0), H=n_heads, KVH=n_kv, KO=n_out_kv,
                DH=head_dim, BH=triton.next_power_of_2(n_heads),
                BKO=triton.next_power_of_2(n_out_kv), LORA_Q=has_lq, LORA_V=has_lv, num_warps=4)
        return dqkv, dlq, dlv, None, None, None, None, None, None, None

"""RMSNorm on the fp32 residual stream, fused with the fp16 casts and the LoRA input dropout.

Base Soup (Qwen2RMSNorm with fp32 weight from prepare_model_for_kbit_training) does, per norm:
    x.to(fp32) -> pow(2) -> mean -> +eps -> rsqrt -> x*r -> .to(fp32) -> w*(...)   [fp32 out]
and then every consumer casts again: bitsandbytes casts the norm output to fp16 for q/k/v (or
gate/up), and each LoRA branch runs dropout(fp32) and autocast casts the result to fp16.
That is ~10 element-wise launches and several fp32 [tokens, 1536] tensors per norm, most of them
kept alive for backward.

Here one program per token row computes, in fp32 and in the same operation order,
    y   = w * (x * rsqrt(mean(x^2) + eps))
and writes only what the next GEMMs read:
    y16           = fp16(y)                    -> base q|k|v (or gate|up) GEMM
    yq16, yv16    = fp16(y * keep * 1/(1-p))   -> LoRA A of q_proj / v_proj (optional)
The two dropout masks come from torch's own dropout RNG (same shapes, same order as base Soup,
so the masks are identical); the kernel reads them through a packed->padded row map and stores
them as bits (1/8 of a bool mask) for backward. Backward recomputes x*r from x and the saved
rstd, sums the three incoming fp16 gradients in fp32 and applies the RMSNorm backward.
"""

import torch
import triton
import triton.language as tl

from .offload import Saved, offload_resid


@triton.jit
def rms_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, eps,
                   mq_ptr, mv_ptr, row_map_ptr, yq_ptr, yv_ptr, bq_ptr, bv_ptr, drop_scale,
                   D: tl.constexpr, BLOCK_D: tl.constexpr, DROPOUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    valid = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=valid, other=0.0)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0)
    var = tl.sum(x * x, axis=0) / D
    r = tl.rsqrt(var + eps)
    y = w * (x * r)
    tl.store(rstd_ptr + row, r)
    tl.store(y_ptr + row * D + cols, y.to(tl.float16), mask=valid)
    if DROPOUT:
        src = tl.load(row_map_ptr + row).to(tl.int64)
        keep_q = tl.load(mq_ptr + src * D + cols, mask=valid, other=0).to(tl.int32)
        keep_v = tl.load(mv_ptr + src * D + cols, mask=valid, other=0).to(tl.int32)
        scaled = y * drop_scale
        tl.store(yq_ptr + row * D + cols, tl.where(keep_q != 0, scaled, 0.0).to(tl.float16),
                 mask=valid)
        tl.store(yv_ptr + row * D + cols, tl.where(keep_v != 0, scaled, 0.0).to(tl.float16),
                 mask=valid)
        shifts = tl.arange(0, 8)[None, :]
        nbytes: tl.constexpr = BLOCK_D // 8
        byte_cols = tl.arange(0, nbytes)
        byte_q = tl.sum(tl.reshape(keep_q, (nbytes, 8)) << shifts, axis=1)
        byte_v = tl.sum(tl.reshape(keep_v, (nbytes, 8)) << shifts, axis=1)
        tl.store(bq_ptr + row * (D // 8) + byte_cols, byte_q.to(tl.uint8), mask=byte_cols < D // 8)
        tl.store(bv_ptr + row * (D // 8) + byte_cols, byte_v.to(tl.uint8), mask=byte_cols < D // 8)


@triton.jit
def rms_bwd_kernel(x_ptr, w_ptr, rstd_ptr, dy_ptr, dx_ptr,
                   dyq_ptr, dyv_ptr, bq_ptr, bv_ptr, drop_scale,
                   D: tl.constexpr, BLOCK_D: tl.constexpr, DROPOUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    valid = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=valid, other=0.0)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0)
    r = tl.load(rstd_ptr + row)
    dy = tl.load(dy_ptr + row * D + cols, mask=valid, other=0.0).to(tl.float32)
    if DROPOUT:
        nbytes: tl.constexpr = BLOCK_D // 8
        byte_cols = tl.arange(0, nbytes)
        shifts = tl.arange(0, 8)[None, :]
        byte_q = tl.load(bq_ptr + row * (D // 8) + byte_cols, mask=byte_cols < D // 8, other=0)
        byte_v = tl.load(bv_ptr + row * (D // 8) + byte_cols, mask=byte_cols < D // 8, other=0)
        keep_q = tl.reshape((byte_q.to(tl.int32)[:, None] >> shifts) & 1, (BLOCK_D,))
        keep_v = tl.reshape((byte_v.to(tl.int32)[:, None] >> shifts) & 1, (BLOCK_D,))
        dyq = tl.load(dyq_ptr + row * D + cols, mask=valid, other=0.0).to(tl.float32)
        dyv = tl.load(dyv_ptr + row * D + cols, mask=valid, other=0.0).to(tl.float32)
        dy = dy + tl.where(keep_q != 0, dyq * drop_scale, 0.0)
        dy = dy + tl.where(keep_v != 0, dyv * drop_scale, 0.0)
    g = dy * w
    xhat_dot = tl.sum(g * x, axis=0) / D
    dx = r * g - x * (r * r * r) * xhat_dot
    tl.store(dx_ptr + row * D + cols, dx, mask=valid)


class FusedRMSNorm(torch.autograd.Function):
    """x (fp32 [N, D]) -> fp16 y, and with masks also the two dropped LoRA inputs."""

    @staticmethod
    def forward(ctx, x, weight, eps, mask_q, mask_v, row_map, p):
        n, d = x.shape
        block_d = triton.next_power_of_2(d)
        dropout = mask_q is not None
        y = torch.empty(n, d, dtype=torch.float16, device=x.device)
        rstd = torch.empty(n, dtype=torch.float32, device=x.device)
        scale = 1.0 / (1.0 - p) if dropout else 1.0
        if dropout:
            yq, yv = torch.empty_like(y), torch.empty_like(y)
            bits_q = torch.empty(n, d // 8, dtype=torch.uint8, device=x.device)
            bits_v = torch.empty_like(bits_q)
            args = (mask_q, mask_v, row_map, yq, yv, bits_q, bits_v)
        else:
            yq = yv = bits_q = bits_v = None
            args = (y, y, rstd, y, y, y, y)  # unused pointers
        if n:
            rms_fwd_kernel[(n,)](x, weight, y, rstd, eps, *args, scale, D=d, BLOCK_D=block_d,
                                 DROPOUT=dropout, num_warps=8 if d > 2048 else 4)
        ctx.dropout, ctx.scale = dropout, scale
        # x is only needed if a gradient flows back into it (not for the embedding output)
        ctx.x = Saved(x, offload_resid()) if x.requires_grad else None
        if dropout:
            ctx.save_for_backward(weight, rstd, bits_q, bits_v)
            return y, yq, yv
        ctx.save_for_backward(weight, rstd)
        return y

    @staticmethod
    def backward(ctx, dy, *rest):
        if ctx.x is None:
            return None, None, None, None, None, None, None
        if ctx.dropout:
            weight, rstd, bits_q, bits_v = ctx.saved_tensors
            dyq, dyv = rest
        else:
            weight, rstd = ctx.saved_tensors
        x = ctx.x.get()
        ctx.x = None
        n, d = x.shape
        dx = torch.empty_like(x)
        if dy is None:
            dy = torch.zeros(n, d, dtype=torch.float16, device=x.device)
        dy = dy.contiguous()
        if ctx.dropout:
            dyq = torch.zeros_like(dy) if dyq is None else dyq.contiguous()
            dyv = torch.zeros_like(dy) if dyv is None else dyv.contiguous()
            args = (dyq, dyv, bits_q, bits_v)
        else:
            args = (dy, dy, dy, dy)
        if n:
            rms_bwd_kernel[(n,)](x, weight, rstd, dy, dx, *args, ctx.scale, D=d,
                                 BLOCK_D=triton.next_power_of_2(d), DROPOUT=ctx.dropout,
                                 num_warps=8 if d > 2048 else 4)
        return dx, None, None, None, None, None, None


def rmsnorm_fp16(x, weight, eps):
    return FusedRMSNorm.apply(x, weight, eps, None, None, None, 0.0)


def rmsnorm_fp16_dropout(x, weight, eps, mask_q, mask_v, row_map, p):
    return FusedRMSNorm.apply(x, weight, eps, mask_q, mask_v, row_map, p)

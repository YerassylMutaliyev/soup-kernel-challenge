"""The training forward of Qwen2 + 4-bit LoRA, rebuilt around our Triton kernels.

What happens to one batch (B rows, right-padded to L):

1. Unpadding: only the N real tokens (attention_mask == 1) are computed. The pad rows of base
   Soup never reach the loss, the logged entropy/accuracy or the rewind log, so skipping them
   changes no output; every matmul and element-wise op then works on N instead of B*L rows.
   The LoRA dropout masks are still drawn by torch for the full [B*L, D] shape, in the same
   order (q then v, layer by layer), so they are the exact masks base Soup draws.
2. Per decoder layer (fp32 residual stream, like base Soup):
     RMSNorm + fp16 casts + both LoRA dropouts   -> k_rmsnorm (one launch)
     q|k|v projection                            -> NF4 decode (k_nf4) + ONE cuBLAS GEMM + bias
     LoRA A/B on q and v                         -> torch (tiny r=16 GEMMs, the trained part)
     LoRA merge + RoPE + GQA expand + fp16       -> k_rope (one launch)
     causal attention per sequence               -> torch SDPA (memory-efficient kernel)
     o projection, residual                      -> NF4 decode + cuBLAS
     RMSNorm + fp16 cast                         -> k_rmsnorm
     gate|up GEMM, SiLU*up, down GEMM            -> NF4 decode + cuBLAS + k_glu + cuBLAS
3. Final RMSNorm -> fused head (k_head): loss, entropy, accuracy, per-row loss, no full logits.

Frozen 4-bit linears never save their input (there is no weight gradient), the MLP saves only
its fp16 gate|up output, and attention saves what SDPA needs.
"""

import os
import time

import torch
import torch.nn.functional as F

from . import trace

from .k_glu import glu_backward, glu_forward
from .k_head import FusedLMHeadLoss
from .k_nf4 import NF4Weight, StackedNF4
from .k_rmsnorm import rmsnorm_fp16, rmsnorm_fp16_dropout
from .k_rope import FusedRopeQKV
from .offload import (OFFLOADER, Saved, keep_last_layers, offload_attn, offload_mlp,
                      offload_resid, pack_saved, unpack_saved)

HEAD_CHUNK = int(os.environ.get("SOUP_KERNELS_HEAD_CHUNK", "256"))  # logits rows alive at once


class FrozenNF4Linear(torch.autograd.Function):
    """y16 = x16 @ W^T (+ b16) with W decoded from NF4 on the fly, forward and backward."""

    @staticmethod
    def forward(ctx, x16, stacked, bias16):
        with torch.autocast("cuda", enabled=False):
            w = stacked.decode()
            if bias16 is not None:
                y = torch.addmm(bias16, x16, w.t())
            else:
                y = torch.mm(x16, w.t())
        ctx.stacked = stacked
        return y

    @staticmethod
    def backward(ctx, dy):
        with torch.autocast("cuda", enabled=False):
            w = ctx.stacked.decode()
            dx = torch.mm(dy.to(torch.float16), w)
        return dx, None, None


class FrozenNF4GatedMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) with gate|up as one GEMM; saves only the fp16 gate|up."""

    @staticmethod
    def forward(ctx, x16, gate_up, down):
        with torch.autocast("cuda", enabled=False):
            w_gu = gate_up.decode()
            gu = torch.mm(x16, w_gu.t())
            del w_gu
            h = glu_forward(gu)
            w_d = down.decode()
            y = torch.mm(h, w_d.t())
        ctx.gu = Saved(gu, offload_mlp())
        del gu
        ctx.gate_up, ctx.down = gate_up, down
        return y

    @staticmethod
    def backward(ctx, dy):
        gu = ctx.gu.get()
        ctx.gu = None
        with torch.autocast("cuda", enabled=False):
            w_d = ctx.down.decode()
            dh = torch.mm(dy.to(torch.float16), w_d)
            del w_d
            dgu = glu_backward(gu, dh, out=gu)  # in place: gu is not needed afterwards
            del dh, gu
            w_gu = ctx.gate_up.decode()
            dx = torch.mm(dgu, w_gu)
        return dx, None, None


class LoRABranch(torch.autograd.Function):
    """peft's vanilla LoRA under fp16 autocast: (x16 @ A^T) @ B^T * scaling, all fp16, with the
    fp32 parameters' gradients. Written out so its input can be offloaded like ours."""

    @staticmethod
    def forward(ctx, x16, a_weight, b_weight, scaling):
        with torch.autocast("cuda", enabled=False):
            a16 = a_weight.to(torch.float16)
            b16 = b_weight.to(torch.float16)
            h = torch.mm(x16, a16.t())
            out = torch.mm(h, b16.t()) * scaling
        ctx.x = Saved(x16, offload_resid())
        ctx.save_for_backward(h, a16, b16)
        ctx.scaling = scaling
        ctx.needs_x = x16.requires_grad
        return out

    @staticmethod
    def backward(ctx, dout):
        h, a16, b16 = ctx.saved_tensors
        x16 = ctx.x.get()
        ctx.x = None
        with torch.autocast("cuda", enabled=False):
            g = dout.to(torch.float16) * ctx.scaling
            db = torch.mm(g.t(), h)
            dh = torch.mm(g, b16)
            da = torch.mm(dh.t(), x16)
            dx = torch.mm(dh, a16) if ctx.needs_x else None
        return dx, da.float(), db.float(), None


def _base_of(module):
    return getattr(module, "base_layer", module)


def _is_nf4_linear(module):
    try:
        import bitsandbytes as bnb
    except Exception:  # noqa: BLE001
        return False
    base = _base_of(module)
    if not isinstance(base, bnb.nn.Linear4bit):
        return False
    qs = getattr(base.weight, "quant_state", None)
    return qs is not None and qs.quant_type == "nf4" and len(qs.shape) == 2


def _lora_of(module):
    """(adapter name, dropout p) of a vanilla peft LoRA wrapper, None if it has no LoRA,
    False if it has one we do not handle."""
    if not hasattr(module, "lora_A"):
        return None
    if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
        return False
    active = [a for a in getattr(module, "active_adapters", []) if a in module.lora_A]
    if not active:
        return None
    if len(active) != 1:
        return False
    name = active[0]
    if name in getattr(module, "lora_variant", {}) or getattr(module, "use_dora", {}).get(name):
        return False
    if getattr(module.lora_A[name], "bias", None) is not None or \
            getattr(module.lora_B[name], "bias", None) is not None:
        return False
    drop = module.lora_dropout[name]
    p = float(getattr(drop, "p", 0.0)) if isinstance(drop, torch.nn.Dropout) else 0.0
    if not isinstance(drop, (torch.nn.Dropout, torch.nn.Identity)):
        return False
    return name, p


class LayerPlan:
    """The frozen parts of one decoder layer, decoded once into kernel arguments."""

    def __init__(self, layer, config):
        attn, mlp = layer.self_attn, layer.mlp
        q, k, v, o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        for mod in (q, k, v, o, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if not _is_nf4_linear(mod):
                raise ValueError(f"not an NF4 linear: {type(_base_of(mod)).__name__}")
        for mod in (k, o, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if _lora_of(mod) is not None:
                raise ValueError("LoRA outside q/v is not handled")
        self.q, self.v = q, v
        self.lora_q, self.lora_v = _lora_of(q), _lora_of(v)
        if self.lora_q is False or self.lora_v is False:
            raise ValueError("unsupported LoRA")
        ps = {lo[1] for lo in (self.lora_q, self.lora_v) if lo}
        if len(ps) > 1:
            raise ValueError("different LoRA dropouts")
        self.p = ps.pop() if ps else 0.0
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.n_heads
        self.scaling = attn.scaling
        self.qkv = StackedNF4([NF4Weight(_base_of(m).weight) for m in (q, k, v)])
        self.o = StackedNF4([NF4Weight(_base_of(o).weight)])
        self.gate_up = StackedNF4([NF4Weight(_base_of(m).weight)
                                   for m in (mlp.gate_proj, mlp.up_proj)])
        self.down = StackedNF4([NF4Weight(_base_of(mlp.down_proj).weight)])
        biases = [_base_of(m).bias for m in (q, k, v)]
        if any(b is None for b in biases):
            if any(b is not None for b in biases):
                raise ValueError("mixed q/k/v biases")
            self.qkv_bias = None
        else:
            self.qkv_bias = torch.cat([b.detach().to(torch.float16) for b in biases])
        if _base_of(o).bias is not None or any(_base_of(m).bias is not None for m in
                                               (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            raise ValueError("unexpected bias")
        self.ln1, self.ln2 = layer.input_layernorm, layer.post_attention_layernorm

    def lora_out(self, module, lora, x16):
        name = lora[0]
        return LoRABranch.apply(x16, module.lora_A[name].weight, module.lora_B[name].weight,
                                float(module.scaling[name]))


class ModelPlan:
    def __init__(self, lm):
        model = lm.model
        cfg = lm.config
        if getattr(cfg, "sliding_window", None) and getattr(cfg, "use_sliding_window", False):
            raise ValueError("sliding window attention")
        if getattr(model, "has_sliding_layers", False):
            raise ValueError("sliding layers")
        if any(t != "full_attention" for t in getattr(cfg, "layer_types", ["full_attention"])):
            raise ValueError("non-full attention layers")
        if getattr(cfg, "hidden_act", "silu") != "silu":
            raise ValueError("activation")
        self.layers = [LayerPlan(layer, cfg) for layer in model.layers[:cfg.num_hidden_layers]]
        self.embed = model.embed_tokens
        self.norm = model.norm
        self.rotary = model.rotary_emb
        head = lm.lm_head
        if head.weight.data_ptr() != self.embed.weight.data_ptr() or head.bias is not None:
            raise ValueError("untied or biased head")
        if head.weight.requires_grad or self.embed.weight.requires_grad:
            raise ValueError("trainable head/embedding")
        self.gqa_sdpa = False


def _keep_mask(rows, cols, p, device):
    """The keep-mask torch's dropout draws for an fp32 [rows, cols] tensor: same kernel, same
    RNG offsets as nn.Dropout on the [B, L, cols] LoRA input of base Soup."""
    dummy = torch.empty(rows, cols, dtype=torch.float32, device=device)
    _, keep = torch.native_dropout(dummy, p, True)
    return keep


def _attention(q, k, v, cu, scale, gqa):
    """Causal attention per sequence. What SDPA keeps for backward (q, k, v, output) is offloaded
    like our own saved tensors."""
    if offload_attn():
        with torch.autograd.graph.saved_tensors_hooks(pack_saved, unpack_saved):
            return _attention_impl(q, k, v, cu, scale, gqa)
    return _attention_impl(q, k, v, cu, scale, gqa)


def _attention_impl(q, k, v, cu, scale, gqa):
    outs = []
    lens = [e - s for s, e in zip(cu[:-1], cu[1:])]
    if len(set(lens)) == 1 and lens[0] > 0:
        b, n = len(lens), lens[0]
        qb = q.view(b, n, *q.shape[1:]).transpose(1, 2)
        kb = k.view(b, n, *k.shape[1:]).transpose(1, 2)
        vb = v.view(b, n, *v.shape[1:]).transpose(1, 2)
        out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, scale=scale,
                                             enable_gqa=gqa)
        return out.transpose(1, 2).reshape(b * n, -1)
    for s, e in zip(cu[:-1], cu[1:]):
        if e == s:
            continue
        qb = q[s:e].unsqueeze(0).transpose(1, 2)
        kb = k[s:e].unsqueeze(0).transpose(1, 2)
        vb = v[s:e].unsqueeze(0).transpose(1, 2)
        out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, scale=scale,
                                             enable_gqa=gqa)
        outs.append(out.transpose(1, 2).reshape(e - s, -1))
    return torch.cat(outs, 0)


def _layer_forward(plan, x, step):
    dev = x.device
    if plan.p > 0 and (plan.lora_q or plan.lora_v):
        mask_q = _keep_mask(step["padded_rows"], x.shape[1], plan.p, dev) if plan.lora_q else None
        mask_v = _keep_mask(step["padded_rows"], x.shape[1], plan.p, dev) if plan.lora_v else None
        if mask_q is None or mask_v is None:
            raise RuntimeError("LoRA on only one of q/v with dropout is not handled")
        xn, xq, xv = rmsnorm_fp16_dropout(x, plan.ln1.weight, plan.ln1.variance_epsilon,
                                          mask_q, mask_v, step["row_map"], plan.p)
    else:
        xn = rmsnorm_fp16(x, plan.ln1.weight, plan.ln1.variance_epsilon)
        xq = xv = xn
    qkv = FrozenNF4Linear.apply(xn, plan.qkv, plan.qkv_bias)
    lq = plan.lora_out(plan.q, plan.lora_q, xq) if plan.lora_q else None
    lv = plan.lora_out(plan.v, plan.lora_v, xv) if plan.lora_v else None
    n_out_kv = plan.n_kv if step["gqa"] else plan.n_heads
    q, k, v = FusedRopeQKV.apply(qkv, lq, lv, step["pos"], step["cos"], step["sin"],
                                 plan.n_heads, plan.n_kv, n_out_kv, plan.head_dim)
    attn = _attention(q, k, v, step["cu"], plan.scaling, step["gqa"])
    o = FrozenNF4Linear.apply(attn, plan.o, None)
    x = x + o
    xn2 = rmsnorm_fp16(x, plan.ln2.weight, plan.ln2.variance_epsilon)
    y = FrozenNF4GatedMLP.apply(xn2, plan.gate_up, plan.down)
    return x + y


def fused_loss_forward(lm, plan, input_ids, attention_mask, labels, num_items_in_batch,
                       grad_scale):
    """Loss of one labelled batch plus what TRL / Soup log; None if the batch is not one we
    handle (then the caller runs the original forward)."""
    from . import warmup

    t_enter = time.perf_counter()
    warmup.wait()  # the background compile must be done before the first launch
    t_ready = time.perf_counter()
    OFFLOADER.begin_step()
    dev = input_ids.device
    bsz, seqlen = input_ids.shape
    if attention_mask is None:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        mask = attention_mask.to(torch.bool)
    lens = mask.sum(1)
    positions = torch.arange(seqlen, device=dev)
    if not torch.equal(mask, positions[None, :] < lens[:, None]):
        return None  # not right-padded
    lens_list = lens.tolist()
    cu = [0]
    for n in lens_list:
        cu.append(cu[-1] + n)
    flat_idx = mask.reshape(-1).nonzero().squeeze(1)
    pos = (flat_idx % seqlen).to(torch.int32)
    ids = input_ids.reshape(-1).index_select(0, flat_idx)

    with torch.autocast("cuda", enabled=False):
        x = F.embedding(ids, plan.embed.weight).float()
        cos, sin = plan.rotary(x[:1], positions[None])
    cos = cos[0].float().contiguous()
    sin = sin[0].float().contiguous()
    step = {"padded_rows": bsz * seqlen, "row_map": flat_idx, "pos": pos, "cos": cos,
            "sin": sin, "cu": cu, "gqa": plan.gqa_sdpa}
    last_offloaded = len(plan.layers) - keep_last_layers()
    try:
        for i, layer_plan in enumerate(plan.layers):
            OFFLOADER.active = i < last_offloaded
            OFFLOADER.begin_layer()
            x = _layer_forward(layer_plan, x, step)
    finally:
        OFFLOADER.active = True
    OFFLOADER.active = False
    OFFLOADER.begin_layer()
    try:
        h16 = rmsnorm_fp16(x, plan.norm.weight, plan.norm.variance_epsilon)
    finally:
        OFFLOADER.active = True
    t_layers = time.perf_counter()

    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    targets = shifted.reshape(-1).index_select(0, flat_idx).contiguous()
    labelled = (targets != -100)
    targets = torch.where(labelled, targets, torch.full_like(targets, -1))
    if num_items_in_batch is None:
        n_items = labelled.sum().clamp(min=1).float()
    elif torch.is_tensor(num_items_in_batch):
        n_items = num_items_in_batch.to(dev)
    else:
        n_items = torch.tensor(float(num_items_in_batch), device=dev)
    loss, ent, tok_loss, correct = FusedLMHeadLoss.apply(
        h16, plan.embed.weight, targets, grad_scale, n_items, HEAD_CHUNK)
    if trace.on():  # the head synchronizes once (labelled row count), so this is GPU time too
        trace.log(ev="fwd", tokens=cu[-1], padded=bsz * seqlen, wait=round(t_ready - t_enter, 4),
                  layers_cpu=round(t_layers - t_ready, 4),
                  head=round(time.perf_counter() - t_layers, 4), **trace.memory())

    token_entropy = torch.zeros(bsz * seqlen, dtype=torch.float32, device=dev)
    token_entropy.index_copy_(0, flat_idx, ent)
    token_loss = torch.zeros(bsz * seqlen, dtype=torch.float32, device=dev)
    token_loss.index_copy_(0, flat_idx, tok_loss)
    stats = {"token_entropy": token_entropy.view(bsz, seqlen),
             "token_loss": token_loss.view(bsz, seqlen),
             "correct": correct, "labelled": labelled.sum()}
    return loss, stats

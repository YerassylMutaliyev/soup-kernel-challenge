"""Fused LM head: cross-entropy, entropy, argmax accuracy and the logits gradient, chunked.

Base Soup materializes the logits of the whole batch ([B*L, 151936]) several times per step:
fp16 out of the head, an fp32 copy for the loss, the fp32 log-softmax kept for backward, an fp32
copy made by accelerate's output conversion, TRL's entropy and accuracy passes, the rewind log's
per-row cross-entropy and two fp32 gradients of the same size in backward. At 2048 tokens that is
several GB and ~7% of the step.

Here the head runs over `chunk` token rows at a time (labelled rows first). For each chunk
cuBLAS computes the fp16 logits, then ONE Triton program per row streams the row twice:
  pass 1  online max / sum(exp) / sum(exp*z) and argmax  ->  logsumexp, entropy, CE, hit
  pass 2  dlogits = (softmax - onehot) * (grad_scale / n_items), written over the logits in
          fp16 (the same values base Soup's fp16 logits gradient has, including the GradScaler
          factor, so small probabilities do not underflow differently)
and cuBLAS turns the chunk's dlogits into the hidden-state gradient (labelled rows only: the
prompt rows have no label and a zero gradient, base Soup multiplies their zeros anyway). The
logits never exist for more than one chunk, and the backward pass only rescales the stored
hidden gradient.

Idea credit: the online-softmax single-pass loss and in-place gradient follow the approach of
Liger-Kernel's fused linear cross-entropy (https://github.com/linkedin/Liger-Kernel) and
Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018); the code is ours, and
it also produces the entropy / accuracy / per-row losses that Soup and TRL log.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 2048


@triton.jit(do_not_specialize=["write_grad"])
def lm_head_ce_kernel(logits_ptr, stride_row, target_ptr, gscale_ptr,
                      ent_ptr, loss_ptr, hit_ptr, write_grad, V: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    target = tl.load(target_ptr + row)
    running_max = float("-inf")
    sum_exp = 0.0
    sum_exp_z = 0.0
    best_val = float("-inf")
    best_idx = 0
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        valid = cols < V
        z = tl.load(base + cols, mask=valid, other=float("-inf")).to(tl.float32)
        block_max = tl.max(z, axis=0)
        block_arg = tl.argmax(z, axis=0)
        best_idx = tl.where(block_max > best_val, start + block_arg, best_idx)
        best_val = tl.maximum(best_val, block_max)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        e = tl.exp(z - new_max)
        sum_exp = sum_exp * rescale + tl.sum(e, axis=0)
        sum_exp_z = sum_exp_z * rescale + tl.sum(tl.where(valid, e * z, 0.0), axis=0)
        running_max = new_max
    lse = running_max + tl.log(sum_exp)
    tl.store(ent_ptr + row, lse - sum_exp_z / sum_exp)
    labelled = target >= 0
    safe_target = tl.where(labelled, target, 0)
    z_target = tl.load(base + safe_target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
    tl.store(hit_ptr + row, ((best_idx == target) & labelled).to(tl.int32))
    if write_grad != 0:  # a runtime flag: one compiled kernel for both kinds of chunk
        gs = tl.load(gscale_ptr)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            valid = cols < V
            z = tl.load(base + cols, mask=valid, other=0.0).to(tl.float32)
            g = tl.exp(z - lse) * gs
            g = tl.where(cols == target, g - gs, g)
            g = tl.where(labelled, g, 0.0)
            tl.store(base + cols, g.to(tl.float16), mask=valid)


class FusedLMHeadLoss(torch.autograd.Function):
    """loss = sum(CE over labelled rows) / n_items, plus per-row entropy, CE and argmax hits."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, grad_scale, n_items, chunk):
        with torch.autocast("cuda", enabled=False):
            n, d = hidden.shape
            vocab = weight.shape[0]
            dev = hidden.device
            labelled = targets >= 0
            order = torch.argsort((~labelled).to(torch.int8), stable=True)
            n_lab = int(labelled.sum())
            h_sorted = hidden.index_select(0, order)
            t_sorted = targets.index_select(0, order).contiguous()
            ent = torch.empty(n, dtype=torch.float32, device=dev)
            tok_loss = torch.empty(n, dtype=torch.float32, device=dev)
            hits = torch.empty(n, dtype=torch.int32, device=dev)
            gs = (grad_scale.float() / n_items).reshape(1).to(device=dev, dtype=torch.float32)
            dh_lab = torch.empty(n_lab, d, dtype=torch.float16, device=dev)
            buf = torch.empty(min(chunk, max(n, 1)), vocab, dtype=torch.float16, device=dev)
            w_t = weight.t()
            for c0 in range(0, n, chunk):
                c1 = min(n, c0 + chunk)
                logits = buf[:c1 - c0]
                torch.mm(h_sorted[c0:c1], w_t, out=logits)
                write_grad = c0 < n_lab
                lm_head_ce_kernel[(c1 - c0,)](
                    logits, logits.stride(0), t_sorted[c0:c1], gs, ent[c0:c1], tok_loss[c0:c1],
                    hits[c0:c1], int(write_grad), V=vocab, BLOCK=BLOCK_V, num_warps=8)
                if write_grad:
                    r = min(c1, n_lab) - c0
                    torch.mm(logits[:r], weight, out=dh_lab[c0:c0 + r])
            del buf
            loss_sum = tok_loss[:n_lab].sum()
            loss = loss_sum / n_items
            # back to the packed token order
            ent_out = torch.empty_like(ent).index_copy_(0, order, ent)
            loss_out = torch.empty_like(tok_loss).index_copy_(0, order, tok_loss)
            correct = hits.sum()
            dh = torch.zeros(n, d, dtype=torch.float16, device=dev)
            if n_lab:
                dh.index_copy_(0, order[:n_lab], dh_lab)
        ctx.save_for_backward(dh, grad_scale.detach().float().reshape(()).to(dev).clone())
        ctx.mark_non_differentiable(ent_out, loss_out, correct)
        return loss, ent_out, loss_out, correct

    @staticmethod
    def backward(ctx, grad_loss, *unused):
        dh, scale_used = ctx.saved_tensors
        ratio = (grad_loss.float() / scale_used).to(torch.float16)
        return dh * ratio, None, None, None, None, None

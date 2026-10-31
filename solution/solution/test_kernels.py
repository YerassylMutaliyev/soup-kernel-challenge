"""Our kernels against plain PyTorch (and bitsandbytes), forward and backward.

On a GPU: `python -m pytest solution -q`. Without a GPU the same tests run through Triton's
interpreter: `TRITON_INTERPRET=1 SOUP_KERNELS_TEST_CPU=1 python -m pytest solution -q`.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

CPU = os.environ.get("SOUP_KERNELS_TEST_CPU") == "1"
DEV = "cpu" if CPU else "cuda"
pytestmark = pytest.mark.skipif(not CPU and not torch.cuda.is_available(),
                                reason="needs a CUDA GPU (or SOUP_KERNELS_TEST_CPU=1)")


def close(a, b, tol):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


# ----------------------------------------------------------------------------------- NF4

@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(256, 1536), (64, 320)])
def test_nf4_decode_matches_bitsandbytes(nested, shape):
    import bitsandbytes.functional as bnbf

    from solution.k_nf4 import NF4Weight, StackedNF4

    torch.manual_seed(0)
    w = (torch.randn(shape, device=DEV) * 0.02).to(torch.float16)
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                       compress_statistics=nested)
    ref = bnbf.dequantize_4bit(packed, state).to(torch.float16)

    class P:  # what NF4Weight reads from a Params4bit
        data = packed
        quant_state = state

    out = StackedNF4([NF4Weight(P()), NF4Weight(P())]).decode()
    assert torch.equal(out[:shape[0]], ref)
    assert torch.equal(out[shape[0]:], ref)


# ----------------------------------------------------------------------------------- RMSNorm

def _ref_norm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w * (xf * torch.rsqrt(var + eps))


@pytest.mark.parametrize("dropout", [True, False])
def test_rmsnorm(dropout):
    from solution.k_rmsnorm import rmsnorm_fp16, rmsnorm_fp16_dropout

    torch.manual_seed(0)
    n, d, padded, p, eps = 37, 1536, 50, 0.05, 1e-6
    x = torch.randn(n, d, device=DEV, requires_grad=True)
    w = (1 + 0.1 * torch.randn(d, device=DEV))
    row_map = torch.randperm(padded, device=DEV)[:n].sort().values
    gy = torch.randn(n, d, device=DEV).half()
    if dropout:
        mq = torch.rand(padded, d, device=DEV) > p
        mv = torch.rand(padded, d, device=DEV) > p
        gq, gv = torch.randn(n, d, device=DEV).half(), torch.randn(n, d, device=DEV).half()
        y, yq, yv = rmsnorm_fp16_dropout(x, w, eps, mq, mv, row_map, p)
        (y.float() * gy.float() + yq.float() * gq.float() + yv.float() * gv.float()).sum().backward()
    else:
        y = rmsnorm_fp16(x, w, eps)
        (y.float() * gy.float()).sum().backward()
    dx = x.grad.clone()
    x.grad = None
    ref = _ref_norm(x, w, eps)
    close(y, ref.half(), 1e-3)
    if dropout:
        scale = 1 / (1 - p)
        refq = (ref * mq[row_map] * scale)
        refv = (ref * mv[row_map] * scale)
        close(yq, refq.half(), 1e-3)
        close(yv, refv.half(), 1e-3)
        (ref * gy.float() + refq * gq.float() + refv * gv.float()).sum().backward()
    else:
        (ref * gy.float()).sum().backward()
    close(dx, x.grad, 2e-3)


# ----------------------------------------------------------------------------------- RoPE

def _rope_cos_sin(seqlen, dh, device):
    inv = 1.0 / (1000000.0 ** (torch.arange(0, dh, 2, dtype=torch.float32, device=device) / dh))
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32, device=device), inv)
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("gqa", [False, True])
def test_rope_qkv(gqa):
    from solution.k_rope import FusedRopeQKV

    torch.manual_seed(0)
    n, h, kvh, dh, seqlen = 29, 12, 2, 128, 40
    ko = kvh if gqa else h
    cos, sin = _rope_cos_sin(seqlen, dh, DEV)
    pos = torch.randint(0, seqlen, (n,), device=DEV, dtype=torch.int32)
    qkv = torch.randn(n, (h + 2 * kvh) * dh, device=DEV).half().requires_grad_()
    lq = torch.randn(n, h * dh, device=DEV).half().requires_grad_()
    lv = torch.randn(n, kvh * dh, device=DEV).half().requires_grad_()
    q, k, v = FusedRopeQKV.apply(qkv, lq, lv, pos, cos, sin, h, kvh, ko, dh)
    gq, gk, gv = (torch.randn_like(t) for t in (q, k, v))
    (q.float() * gq.float()).sum().add_((k.float() * gk.float()).sum()).add_(
        (v.float() * gv.float()).sum()).backward()
    grads = [t.grad.clone() for t in (qkv, lq, lv)]
    for t in (qkv, lq, lv):
        t.grad = None

    def rot(x):
        return torch.cat((-x[..., dh // 2:], x[..., :dh // 2]), -1)

    c, s = cos[pos.long()][:, None, :], sin[pos.long()][:, None, :]
    qf = qkv[:, :h * dh].float().view(n, h, dh) + lq.float().view(n, h, dh)
    kf = qkv[:, h * dh:(h + kvh) * dh].float().view(n, kvh, dh)
    vf = qkv[:, (h + kvh) * dh:].float().view(n, kvh, dh) + lv.float().view(n, kvh, dh)
    rq = qf * c + rot(qf) * s
    rk = kf * c + rot(kf) * s
    if not gqa:
        rk = rk.repeat_interleave(h // kvh, dim=1)
        vf = vf.repeat_interleave(h // kvh, dim=1)
    close(q, rq.half(), 2e-3)
    close(k, rk.half(), 2e-3)
    close(v, vf.half(), 2e-3)
    (rq * gq.float()).sum().add_((rk * gk.float()).sum()).add_((vf * gv.float()).sum()).backward()
    for got, t in zip(grads, (qkv, lq, lv)):
        close(got, t.grad, 2e-2)


# ----------------------------------------------------------------------------------- GLU

def test_glu():
    from solution.k_glu import glu_backward, glu_forward

    torch.manual_seed(0)
    m, i = 33, 320
    gu = (torch.randn(m, 2 * i, device=DEV) * 3).half()
    dh = torch.randn(m, i, device=DEV).half()
    h = glu_forward(gu)
    dgu = glu_backward(gu, dh)
    g = gu[:, :i].float().requires_grad_()
    u = gu[:, i:].float().requires_grad_()
    ref = F.silu(g) * u
    close(h, ref.half(), 2e-3)
    ref.backward(dh.float())
    close(dgu[:, :i], g.grad.half(), 2e-3)
    close(dgu[:, i:], u.grad.half(), 2e-3)


# ----------------------------------------------------------------------------------- head

@pytest.mark.parametrize("chunk", [16, 1024])
def test_lm_head(chunk):
    from solution.k_head import FusedLMHeadLoss

    torch.manual_seed(0)
    n, d, vocab = 41, 64, 5000
    h = (torch.randn(n, d, device=DEV) * 0.5).half().requires_grad_()
    w = (torch.randn(vocab, d, device=DEV) * 0.5).half()
    targets = torch.randint(0, vocab, (n,), device=DEV)
    targets[::3] = -1
    n_items = (targets >= 0).sum().float() + 3  # the trainer's count can differ from ours
    scale = torch.tensor(1024.0, device=DEV)
    loss, ent, tok_loss, correct = FusedLMHeadLoss.apply(h, w, targets, scale, n_items, chunk)
    (loss * scale).backward()
    dh = h.grad.clone()
    h.grad = None

    logits = (h @ w.t()).float()
    logp = torch.log_softmax(logits, -1)
    ref_ent = -(logp.exp() * logp).sum(-1)
    lab = targets >= 0
    ref_tok = torch.zeros(n, device=DEV)
    ref_tok[lab] = -logp[lab].gather(1, targets[lab][:, None]).squeeze(1)
    ref_loss = ref_tok.sum() / n_items
    ref_correct = ((logits.argmax(-1) == targets) & lab).sum()
    close(loss, ref_loss, 1e-4)
    close(ent, ref_ent, 1e-3)
    close(tok_loss, ref_tok, 1e-3)
    assert int(correct) == int(ref_correct)
    (ref_loss * scale).backward()
    close(dh, h.grad, 2e-2)


# ----------------------------------------------------------------------------------- GPU-only

gpu_only = pytest.mark.skipif(CPU or not torch.cuda.is_available(), reason="needs CUDA")


@gpu_only
def test_dropout_mask_is_the_one_torch_dropout_draws():
    """fused_qwen2 draws the LoRA dropout masks itself; they must be nn.Dropout's masks."""
    from solution.fused_qwen2 import _keep_mask

    x = torch.randn(4, 77, 1536, device="cuda") + 3.0  # no exact zeros
    torch.manual_seed(123)
    ref = F.dropout(x, 0.05, training=True) != 0
    torch.manual_seed(123)
    keep = _keep_mask(4 * 77, 1536, 0.05, "cuda")
    assert torch.equal(keep.view(4, 77, 1536), ref)
    after_ref = torch.rand(8, device="cuda")
    torch.manual_seed(123)
    F.dropout(x, 0.05, training=True)
    assert torch.equal(after_ref, torch.rand(8, device="cuda"))  # same RNG consumption


@gpu_only
def test_offload_is_exact(monkeypatch):
    """Activations offloaded to pinned host memory come back bit-identical."""
    import bitsandbytes.functional as bnbf

    from solution.fused_qwen2 import FrozenNF4GatedMLP, LoRABranch
    from solution.k_nf4 import NF4Weight, StackedNF4
    from solution.k_rmsnorm import rmsnorm_fp16
    from solution.offload import OFFLOADER

    torch.manual_seed(0)
    d, i, n, layers = 256, 512, 300, 4

    def nf4(shape):
        w = (torch.randn(shape, device="cuda") * 0.02).half()
        packed, st = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                        compress_statistics=True)

        class P:
            data = packed
            quant_state = st
        return NF4Weight(P())

    mlps = [(StackedNF4([nf4((i, d)), nf4((i, d))]), StackedNF4([nf4((d, i))]))
            for _ in range(layers)]
    norms = [1 + 0.1 * torch.randn(d, device="cuda") for _ in range(layers)]
    lora_a = (0.05 * torch.randn(16, d, device="cuda")).requires_grad_()
    lora_b = (0.05 * torch.randn(d, 16, device="cuda")).requires_grad_()
    x0 = torch.randn(n, d, device="cuda", requires_grad=True)

    def run(flag):
        monkeypatch.setenv("SOUP_KERNELS_OFFLOAD_MLP", flag)
        monkeypatch.setenv("SOUP_KERNELS_OFFLOAD_RESID", flag)
        OFFLOADER.begin_step()
        x = x0
        for (gu, dn), w in zip(mlps, norms):
            h = rmsnorm_fp16(x, w, 1e-6)
            h = h + LoRABranch.apply(h, lora_a, lora_b, 2.0)
            x = x + FrozenNF4GatedMLP.apply(h, gu, dn)
        x.float().pow(2).mean().backward()
        grads = [t.grad.clone() for t in (x0, lora_a, lora_b)]
        for t in (x0, lora_a, lora_b):
            t.grad = None
        return x.detach(), grads

    out_a, grads_a = run("0")
    out_b, grads_b = run("1")
    torch.cuda.synchronize()
    assert torch.isfinite(out_a).all() and all(torch.isfinite(g).all() for g in grads_a)
    assert torch.equal(out_a, out_b)
    for ga, gb in zip(grads_a, grads_b):
        assert torch.equal(ga, gb)


@gpu_only
def test_offload_keeps_strides():
    """Views (row slices, transposes) come back with the same values and the same strides."""
    from solution.offload import OFFLOADER, Saved, dense

    OFFLOADER.begin_step()
    big = torch.randn(10, 4, 8, device="cuda")
    for t in (big[3:7].transpose(0, 1), big.permute(2, 0, 1), big[2:9]):
        assert dense(t)
        saved = Saved(t, True)
        assert saved.handle is not None
        back = saved.get()
        torch.cuda.synchronize()
        assert back.stride() == t.stride() and torch.equal(back, t)
    assert not dense(big[:, :2])


@gpu_only
@pytest.mark.parametrize("lens", [(70, 130, 33, 1), (64, 64)])
def test_attention_offload(monkeypatch, lens):
    """SDPA's saved tensors offloaded through saved_tensors_hooks give the same gradients."""
    from solution.fused_qwen2 import _attention
    from solution.offload import OFFLOADER
    from solution.warmup import probe_gqa_sdpa

    torch.manual_seed(0)
    gqa = probe_gqa_sdpa()
    heads, kv, hd = 12, (2 if gqa else 12), 128
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    q0 = torch.randn(cu[-1], heads, hd, device="cuda").half().requires_grad_()
    k0 = torch.randn(cu[-1], kv, hd, device="cuda").half().requires_grad_()
    v0 = torch.randn(cu[-1], kv, hd, device="cuda").half().requires_grad_()
    g = torch.randn(cu[-1], heads * hd, device="cuda").half()

    def run(flag):
        monkeypatch.setenv("SOUP_KERNELS_OFFLOAD_ATTN", flag)
        OFFLOADER.begin_step()
        out = _attention(q0 * 1, k0 * 1, v0 * 1, cu, hd ** -0.5, gqa)
        saved = len(OFFLOADER.handles)
        (out.float() * g.float()).sum().backward()
        grads = [t.grad.clone() for t in (q0, k0, v0)]
        for t in (q0, k0, v0):
            t.grad = None
        return out.detach(), grads, saved

    out_a, grads_a, n_a = run("0")
    out_b, grads_b, n_b = run("1")
    torch.cuda.synchronize()
    assert n_a == 0 and n_b > 0
    assert torch.equal(out_a, out_b)
    for ga, gb in zip(grads_a, grads_b):  # SDPA's backward itself is not bit-deterministic
        close(ga, gb, 2e-3)

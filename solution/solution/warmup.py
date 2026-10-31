"""Compile our Triton kernels in background threads while Soup loads the model.

Every measured run starts with an empty Triton cache, so each kernel is compiled inside the
timed `soup train` command. The model load (tokenizer, safetensors, 4-bit quantization) takes
several seconds during which nothing needs our kernels yet, so we compile them then, with exactly
the signatures (dtypes, constexprs) the training step launches them with. Compilation only
(`JITFunction.warmup`) plus their C launchers; different functions compile in parallel threads (each
function stays in one thread). The fused forward waits for all of it before its first launch.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

_THREAD = None
_ERRORS = []
GQA_SDPA = None  # set by the probe below: does SDPA's memory-efficient kernel take GQA?


def probe_gqa_sdpa(n_heads=12, n_kv=2, head_dim=128, device="cuda"):
    """Native GQA (2 KV heads, no 6x copy of K/V) only if the memory-efficient kernel takes it,
    forward and backward; otherwise the RoPE kernel expands the heads."""
    if os.environ.get("SOUP_KERNELS_GQA", "1") != "1" or n_kv == n_heads:
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(1, n_heads, 16, head_dim, device=device, dtype=torch.float16,
                        requires_grad=True)
        k = torch.randn(1, n_kv, 16, head_dim, device=device, dtype=torch.float16,
                        requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                                   enable_gqa=True)
            out.sum().backward()
        torch.cuda.synchronize(device)
        return True
    except Exception:  # noqa: BLE001
        return False


def _groups():
    from . import k_glu, k_head, k_nf4, k_rmsnorm, k_rope

    f16, f32, u8, i32, i64, b1 = (torch.float16, torch.float32, torch.uint8, torch.int32,
                                  torch.int64, torch.bool)
    g = (1,)
    kvo = 2 if GQA_SDPA else 12
    bko = triton_pow2(kvo)
    return [
        # NF4 decode (Soup's 4-bit config uses double quantization)
        [lambda: k_nf4.nf4_unpack_fp16_kernel.warmup(
            u8, u8, f32, f32, 0.5, f32, f16, 2048, BYTES_PER_ABSMAX=32, ABSMAX_PER_SCALE=256,
            NESTED=True, BLOCK=k_nf4.BYTES_PER_PROGRAM, num_warps=4, enable_fp_fusion=False,
            grid=g)],
        [lambda: k_head.lm_head_ce_kernel.warmup(
            f16, 151936, i64, f32, f32, f32, i32, 1, V=151936, BLOCK=k_head.BLOCK_V,
            num_warps=8, grid=g)],
        [lambda: k_rmsnorm.rms_fwd_kernel.warmup(
            f32, f32, f16, f32, 1e-6, b1, b1, i64, f16, f16, u8, u8, 1.05, D=1536, BLOCK_D=2048,
            DROPOUT=True, num_warps=4, grid=g),
         lambda: k_rmsnorm.rms_fwd_kernel.warmup(
            f32, f32, f16, f32, 1e-6, f16, f16, f32, f16, f16, f16, f16, 1.0, D=1536,
            BLOCK_D=2048, DROPOUT=False, num_warps=4, grid=g)],
        [lambda: k_rmsnorm.rms_bwd_kernel.warmup(
            f32, f32, f32, f16, f32, f16, f16, u8, u8, 1.05, D=1536, BLOCK_D=2048,
            DROPOUT=True, num_warps=4, grid=g),
         lambda: k_rmsnorm.rms_bwd_kernel.warmup(
            f32, f32, f32, f16, f32, f16, f16, f16, f16, 1.0, D=1536, BLOCK_D=2048,
            DROPOUT=False, num_warps=4, grid=g)],
        [lambda: k_rope.rope_qkv_fwd_kernel.warmup(
            f16, f16, f16, i32, f32, f32, f16, f16, f16, 2048, H=12, KVH=2, KO=kvo, DH=128,
            BH=16, BKO=bko, LORA_Q=True, LORA_V=True, num_warps=4, grid=g)],
        [lambda: k_rope.rope_qkv_bwd_kernel.warmup(
            f16, f16, f16, i32, f32, f32, f16, f16, f16, 2048, H=12, KVH=2, KO=kvo, DH=128,
            BH=16, BKO=bko, LORA_Q=True, LORA_V=True, num_warps=4, grid=g)],
        [lambda: k_glu.glu_fwd_kernel.warmup(f16, f16, 1000, I=8960, TM=k_glu.TILE_M,
                                             TN=k_glu.TILE_N, num_warps=4, grid=g)],
        [lambda: k_glu.glu_bwd_kernel.warmup(f16, f16, f16, 1000, I=8960, TM=k_glu.TILE_M,
                                             TN=k_glu.TILE_N, num_warps=4, grid=g)],
    ]


def triton_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _run_group(jobs):
    for job in jobs:
        try:
            kernel = job()
            # Also build the kernel's C launcher (a gcc call per signature) and load the cubin
            # now, instead of at the first launch on the training thread.
            init = getattr(kernel, "_init_handles", None)
            if callable(init):
                init()
        except Exception as exc:  # noqa: BLE001 - the real launch compiles it instead
            _ERRORS.append(repr(exc))


def _compile_all():
    global GQA_SDPA
    GQA_SDPA = probe_gqa_sdpa()
    workers = int(os.environ.get("SOUP_KERNELS_WARMUP_THREADS", "4"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_run_group, _groups()))


def start():
    global _THREAD
    if os.environ.get("SOUP_KERNELS_WARMUP", "1") != "1" or not torch.cuda.is_available():
        return
    if _THREAD is None:
        _THREAD = threading.Thread(target=_compile_all, name="soup-kernels-warmup", daemon=True)
        _THREAD.start()


def wait():
    if _THREAD is not None:
        _THREAD.join()

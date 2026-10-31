"""Gated MLP activation on the fused gate|up GEMM output.

The MLP runs gate_proj and up_proj as ONE GEMM against a [2*I, D] fp16 weight decoded by
k_nf4.py, so its output is a single [N, 2*I] fp16 tensor: gate in columns [0, I), up in [I, 2I).
These kernels work on that layout directly (2-D tiles, row stride 2*I):

    forward   h  = fp16( silu(g) * u )                           -> down_proj GEMM
    backward  dg = dh * u * s * (1 + g * (1 - s)),  du = dh * silu(g)  (s = sigmoid(g))
              written back as one [N, 2*I] fp16 tensor so the input gradient is ONE GEMM
              dX = [dg | du] @ [Wg ; Wu].

All math is fp32 on the exact fp16 values base Soup upcasts (bitsandbytes returns fp16 and casts
to fp32), with torch's own silu formula x / (1 + exp(-x)). Base Soup keeps three fp32 [N, I]
tensors per layer for this (gate, silu(gate), up: 6x the bytes of the fp16 gate|up kept here).
"""

import torch
import triton
import triton.language as tl

TILE_M = 16
TILE_N = 256


@triton.jit(do_not_specialize=["M"])
def glu_fwd_kernel(gu_ptr, h_ptr, M, I: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr):
    rows = (tl.program_id(0) * TM + tl.arange(0, TM)).to(tl.int64)[:, None]
    cols = (tl.program_id(1) * TN + tl.arange(0, TN))[None, :]
    ok = (rows < M) & (cols < I)
    g = tl.load(gu_ptr + rows * (2 * I) + cols, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + rows * (2 * I) + I + cols, mask=ok, other=0.0).to(tl.float32)
    act = g / (1.0 + tl.exp(-g))
    tl.store(h_ptr + rows * I + cols, (act * u).to(tl.float16), mask=ok)


@triton.jit(do_not_specialize=["M"])
def glu_bwd_kernel(gu_ptr, dh_ptr, dgu_ptr, M, I: tl.constexpr, TM: tl.constexpr,
                   TN: tl.constexpr):
    rows = (tl.program_id(0) * TM + tl.arange(0, TM)).to(tl.int64)[:, None]
    cols = (tl.program_id(1) * TN + tl.arange(0, TN))[None, :]
    ok = (rows < M) & (cols < I)
    g = tl.load(gu_ptr + rows * (2 * I) + cols, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + rows * (2 * I) + I + cols, mask=ok, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + rows * I + cols, mask=ok, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    act = g / (1.0 + tl.exp(-g))
    d_act = dh * u
    du = dh * act
    dg = d_act * (sig * (1.0 + g * (1.0 - sig)))
    tl.store(dgu_ptr + rows * (2 * I) + cols, dg.to(tl.float16), mask=ok)
    tl.store(dgu_ptr + rows * (2 * I) + I + cols, du.to(tl.float16), mask=ok)


def glu_forward(gu):
    m, two_i = gu.shape
    i = two_i // 2
    h = torch.empty(m, i, dtype=torch.float16, device=gu.device)
    if m:
        grid = (triton.cdiv(m, TILE_M), triton.cdiv(i, TILE_N))
        glu_fwd_kernel[grid](gu, h, m, I=i, TM=TILE_M, TN=TILE_N, num_warps=4)
    return h


def glu_backward(gu, dh, out=None):
    """[dgate | dup]; `out` may be `gu` itself (each program reads its tile before writing it)."""
    m, two_i = gu.shape
    i = two_i // 2
    dgu = torch.empty_like(gu) if out is None else out
    if m:
        grid = (triton.cdiv(m, TILE_M), triton.cdiv(i, TILE_N))
        glu_bwd_kernel[grid](gu, dh, dgu, m, I=i, TM=TILE_M, TN=TILE_N, num_warps=4)
    return dgu

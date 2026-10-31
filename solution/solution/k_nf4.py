"""NF4 -> fp16 dequantization in Triton, bit-exact with bitsandbytes (double-quant aware).

bitsandbytes stores a frozen Linear4bit weight [out, in] as packed bytes (two 4-bit codes per
byte, high nibble first), one absmax per 64 weights, and with double quantization ("nested") the
absmaxes themselves as uint8 codes + one fp32 scale per 256 absmaxes + a global offset.

    w[i] = NF4[code(i)] * absmax[i // 64]
    absmax[b] = CODE2[absmax_q[b]] * absmax2[b // 256] + offset      (nested)

One program decodes BLOCK bytes = 2*BLOCK weights: the nested absmax is decoded in registers
(never materialized), the two nibbles are interleaved with tl.interleave so the fp16 store is
contiguous. FMA contraction is disabled at launch so the absmax is rounded exactly as
bitsandbytes rounds it (multiply, then add), which makes the output identical to
`bitsandbytes.functional.dequantize_4bit`.

Several weights can be decoded into one buffer (q|k|v, gate|up): the caller hands a view of the
destination at the right offset, and one cuBLAS GEMM then covers all of them.
"""

import torch
import triton
import triton.language as tl

BYTES_PER_PROGRAM = 1024


@triton.jit
def nf4_unpack_fp16_kernel(packed_ptr, absq_ptr, abs_code_ptr, abs_scale_ptr, abs_offset,
                           nf4_ptr, out_ptr, n_bytes,
                           BYTES_PER_ABSMAX: tl.constexpr, ABSMAX_PER_SCALE: tl.constexpr,
                           NESTED: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    byte_idx = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = byte_idx < n_bytes
    packed = tl.load(packed_ptr + byte_idx, mask=in_range, other=0).to(tl.int32)
    group = byte_idx // BYTES_PER_ABSMAX
    if NESTED:
        absq = tl.load(absq_ptr + group, mask=in_range, other=0).to(tl.int32)
        code = tl.load(abs_code_ptr + absq, mask=in_range, other=0.0)
        scale = tl.load(abs_scale_ptr + group // ABSMAX_PER_SCALE, mask=in_range, other=0.0)
        absmax = code * scale + abs_offset
    else:
        absmax = tl.load(absq_ptr + group, mask=in_range, other=0.0)
    w_even = tl.load(nf4_ptr + (packed >> 4)) * absmax
    w_odd = tl.load(nf4_ptr + (packed & 15)) * absmax
    w = tl.interleave(w_even, w_odd).to(tl.float16)
    out_idx = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_idx, w, mask=out_idx < 2 * n_bytes)


class NF4Weight:
    """Everything needed to decode one bitsandbytes Params4bit, read once from its quant_state."""

    def __init__(self, param):
        qs = param.quant_state
        if qs.quant_type != "nf4" or qs.dtype not in (torch.float16, torch.float32, None):
            raise ValueError("only nf4 weights are supported")
        self.packed = param.data.view(torch.uint8).reshape(-1)
        self.shape = tuple(qs.shape)
        self.numel = self.shape[0] * self.shape[1]
        if self.numel % 2 or qs.blocksize % 2:
            raise ValueError("odd sizes are not supported")
        self.bytes_per_absmax = qs.blocksize // 2
        self.nested = bool(qs.nested)
        if self.nested:
            self.absq = qs.absmax
            self.abs_code = qs.state2.code.float().contiguous()
            self.abs_scale = qs.state2.absmax.float().contiguous()
            self.absmax_per_scale = qs.state2.blocksize
            self.abs_offset = float(qs.offset)
        else:
            self.absq = qs.absmax.float().contiguous()
            self.abs_code = self.abs_scale = self.absq
            self.absmax_per_scale = 1
            self.abs_offset = 0.0
        self.nf4 = qs.code.float().contiguous().to(self.packed.device)

    def decode_into(self, out):
        n_bytes = self.numel // 2
        grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
        nf4_unpack_fp16_kernel[grid](
            self.packed, self.absq, self.abs_code, self.abs_scale, self.abs_offset, self.nf4, out,
            n_bytes, BYTES_PER_ABSMAX=self.bytes_per_absmax,
            ABSMAX_PER_SCALE=self.absmax_per_scale, NESTED=self.nested, BLOCK=BYTES_PER_PROGRAM,
            num_warps=4, enable_fp_fusion=False)


class StackedNF4:
    """Several NF4 weights with the same input size, decoded into one [sum(out), in] fp16 matrix."""

    def __init__(self, weights):
        self.weights = weights
        self.in_features = weights[0].shape[1]
        if any(w.shape[1] != self.in_features for w in weights):
            raise ValueError("stacked weights need the same input size")
        self.out_features = sum(w.shape[0] for w in weights)

    def decode(self):
        device = self.weights[0].packed.device
        out = torch.empty(self.out_features, self.in_features, dtype=torch.float16, device=device)
        flat = out.view(-1)
        start = 0
        for w in self.weights:
            w.decode_into(flat[start:start + w.numel])
            start += w.numel
        return out

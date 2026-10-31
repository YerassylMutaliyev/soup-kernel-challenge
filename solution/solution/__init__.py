"""Soup Kernel Challenge solution: own Triton kernels for 4-bit LoRA training of Qwen2 in Soup.

`apply()` is called by the harness once, before Soup builds the model and the trainer. It
installs three narrow patches (see patches.py); the training forward/backward of the model then
runs through fused_qwen2.py and these Triton kernels, all written for this task:

    k_nf4.py      nf4_unpack_fp16_kernel   NF4 (double-quant) -> fp16 weights, bit-exact with bnb
    k_rmsnorm.py  rms_fwd_kernel / rms_bwd_kernel
                                           RMSNorm + fp16 casts + both LoRA dropouts (bit-packed
                                           masks), one launch each way
    k_rope.py     rope_qkv_fwd_kernel / rope_qkv_bwd_kernel
                                           LoRA merge + RoPE + GQA expansion + fp16 cast
    k_glu.py      glu_fwd_kernel / glu_bwd_kernel
                                           SiLU(gate)*up on the fused gate|up GEMM output
    k_head.py     lm_head_ce_kernel        chunked LM head: CE, entropy, argmax, dlogits

Set SOUP_KERNELS_DISABLE=1 to turn everything off (base Soup behaviour).
"""

from . import patches


def apply():
    patches.install()
    try:
        from . import warmup

        warmup.start()
    except Exception as exc:  # noqa: BLE001 - compiling ahead is only an optimization
        print(f"[solution] kernel warm-up skipped: {exc}", flush=True)

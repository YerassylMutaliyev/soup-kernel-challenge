# Submission: TEAM_NAME

## Links
- Repository, final commit: FINAL_COMMIT_URL
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): KAGGLE_NOTEBOOK_URL
- Demo video (up to 3 minutes): VIDEO_URL
- Presentation: PRESENTATION_URL

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_unpack_fp16_kernel` | `solution/k_nf4.py` | bitsandbytes NF4 dequantization of all frozen linears (fwd + bwd), stacked q\|k\|v and gate\|up for single GEMMs | SHARE_NF4 % |
| `rms_fwd_kernel`, `rms_bwd_kernel` | `solution/k_rmsnorm.py` | Qwen2RMSNorm + fp16 casts + both LoRA dropouts (bit-packed masks) | SHARE_RMS % |
| `rope_qkv_fwd_kernel`, `rope_qkv_bwd_kernel` | `solution/k_rope.py` | LoRA merge + RoPE + `repeat_kv` + autocast casts | SHARE_ROPE % |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/k_glu.py` | SiLU·up on the fused gate\|up GEMM output (3 fp32 saved tensors → 1 fp16) | SHARE_GLU % |
| `lm_head_ce_kernel` | `solution/k_head.py` | fp32 logits / CE / entropy / accuracy / rewind CE / CE backward over the 151,936-token vocabulary | SHARE_HEAD % |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Fused linear cross-entropy with the gradient in the forward pass — Liger-Kernel,
  https://github.com/linkedin/Liger-Kernel
- Online softmax — Milakov & Gimelshein 2018, https://arxiv.org/abs/1805.02867
- NF4 / double quantization format — QLoRA (Dettmers et al. 2023) and bitsandbytes
- Offloading saved activations to host memory — Unsloth, https://unsloth.ai/blog/long-context

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | RESULT_BASE_TIME | RESULT_SOL_TIME | speedup RESULT_SPEEDUP x |
| Peak GPU memory, MiB | RESULT_BASE_MEM | RESULT_SOL_MEM | base/ours RESULT_MEMRATIO x |
| Eval loss (public rows) | RESULT_BASE_LOSS | RESULT_SOL_LOSS | RESULT_DLOSS % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (padding, fp32 casts, vocabulary-sized logits, fp32
embedding, saved activations), how the kernels work (section 2 of the README), results against
the base (speed, memory, quality), limits and next steps.

## Limitations
See README section 5.

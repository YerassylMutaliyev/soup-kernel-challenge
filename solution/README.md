# Soup Kernel Challenge — a fused Triton training path for 4-bit LoRA on a T4

Decentrathon 6.0, case **Soup Kernel Challenge**. `soup train` (soup-cli 0.77.0) fine-tunes
Qwen2.5-1.5B-Instruct (NF4 4-bit, LoRA r=16 on `q_proj`/`v_proj`) on 256 Kazakh instructions on a
free Kaggle T4. Our `solution/` package replaces the training forward/backward of that run with
our own Triton kernels plus cuBLAS GEMMs, keeps every Soup output (loss curve, entropy, token
accuracy, `rewind.jsonl`, adapter) the same, and cuts time and GPU memory.

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Whole `soup train` command (s) | RESULT_BASE_TIME | RESULT_SOL_TIME | **speedup RESULT_SPEEDUP×** |
| Peak GPU memory, nvidia-smi (MiB) | RESULT_BASE_MEM | RESULT_SOL_MEM | **base/ours RESULT_MEMRATIO×** |
| Eval loss, public rows 40000–40255 | RESULT_BASE_LOSS | RESULT_SOL_LOSS | RESULT_DLOSS % |
| Our kernels' share of training time (check 4) | — | RESULT_SHARE % | ≥ 5 % required |
| `harness/check_rules.py` | — | RESULT_CHECKS | |

Numbers from our Kaggle T4 run: notebook **KAGGLE_NOTEBOOK_URL**, commit **FINAL_COMMIT_URL**
(`runs/result.json` and `runs/checks.json` are in the notebook output).

---

## 1. What we sped up and why

The base step on a T4 (task statement profile): ~44 % element-wise ops and dtype casts, ~36 %
GEMMs, ~7 % softmax over the 151,936-token vocabulary, ~5 % attention, ~5 % NF4 dequantization;
~14.3 GB peak. Reading Soup/peft/bitsandbytes/TRL we found where that comes from:

1. **Padding.** Batches of 4 rows are padded to the longest row; rows average ~220 tokens
   (median 140, max 512), so a large part of every GEMM and element-wise op works on pad tokens
   that never reach the loss or any logged metric.
2. **The fp32 residual stream and its casts.** `prepare_model_for_kbit_training` keeps norms
   and the tied embedding in fp32, so hidden states are fp32. Every 4-bit linear casts its input
   to fp16 and its output back to fp32; each RMSNorm is ~6 launches; each LoRA branch adds a
   dropout and two casts; RoPE + `repeat_kv` are ~20 launches with fp32 copies. Most of these
   fp32 tensors are kept alive for backward: the MLP alone keeps three fp32 `[tokens, 8960]`
   tensors per layer (~6 GB at 2,048 tokens).
3. **The vocabulary.** Logits `[tokens, 151936]` exist in fp16, in fp32 for the loss, as the
   kept log-softmax, again for accelerate's fp32 output conversion, for TRL's entropy and
   accuracy and for the rewind log's per-row losses, plus two fp32 gradients in backward.
4. **The embedding table** is upcast to fp32 (933 MB) although its values are fp16 numbers.
5. **No gradient checkpointing** (a Soup switch we may not touch), so all saved activations of
   28 layers are on the GPU at the end of the forward pass.

## 2. What we built

`solution.apply()` installs three narrow patches (`solution/patches.py`); the labelled training
forward of the model then runs through `solution/fused_qwen2.py`:

```
input_ids [B, L] ──unpad──► N real tokens (B·L − pads)            fp32 residual stream [N, 1536]
 per layer ×28:
   rms_fwd_kernel        RMSNorm → fp16, + both LoRA-dropout inputs (torch's own masks, kept as bits)
   nf4_unpack_fp16 ×3    q|k|v NF4 → one fp16 [2048,1536] matrix  ─► ONE cuBLAS GEMM (+bias)
   LoRA A/B (torch mm)   the only trainable part, fp16 like peft under autocast
   rope_qkv_fwd_kernel   LoRA merge + RoPE + GQA head expansion + fp16, written in SDPA layout
   SDPA per sequence     causal, memory-efficient kernel, no pad keys/queries at all
   nf4_unpack_fp16       o_proj ─► cuBLAS; residual add
   rms_fwd_kernel        RMSNorm → fp16
   nf4_unpack_fp16 ×2    gate|up → one [17920,1536] matrix ─► ONE GEMM → [N, 2·8960] fp16
   glu_fwd_kernel        silu(gate)·up → fp16 ─► down_proj GEMM; residual add
 final rms_fwd_kernel → lm_head_ce_kernel (chunked): loss, entropy, argmax, dlogits in place
```

| Kernel (`@triton.jit`) | File | Replaces in base Soup |
|---|---|---|
| `nf4_unpack_fp16_kernel` | `solution/k_nf4.py` | bitsandbytes dequantize (fwd+bwd of all 196 frozen linears); decodes the double-quantized absmax in registers, writes stacked q\|k\|v and gate\|up weights so one GEMM serves 3 (or 2) projections; **bit-exact** with bnb |
| `rms_fwd_kernel`, `rms_bwd_kernel` | `solution/k_rmsnorm.py` | Qwen2RMSNorm (6 fp32 ops), the fp16 casts in bitsandbytes and autocast, both LoRA dropouts; dropout masks stored as bits (1/8 of a bool mask) |
| `rope_qkv_fwd_kernel`, `rope_qkv_bwd_kernel` | `solution/k_rope.py` | peft's fp32 LoRA add, `apply_rotary_pos_emb`, `repeat_kv`, autocast casts in SDPA; backward sums the expanded KV heads in fp32 |
| `glu_fwd_kernel`, `glu_bwd_kernel` | `solution/k_glu.py` | SiLU·up in fp32 with 3 fp32 saved tensors; works on the fused gate\|up layout and writes `[dgate\|dup]` for one backward GEMM |
| `lm_head_ce_kernel` | `solution/k_head.py` | fp32 logits, CE, TRL entropy and accuracy, rewind per-row CE, CE backward: one row-streaming pass computes logsumexp / entropy / CE / argmax (online softmax), a second writes fp16 dlogits (with the GradScaler factor, exactly like base) |

Around the kernels:

- **Unpadding** (`fused_qwen2.py`): only real tokens are computed; attention runs per sequence
  with `is_causal`. Removing pad tokens inside a batch is explicitly allowed by the rules.
- **Same dropout masks as base.** The LoRA dropout masks are drawn by `torch.native_dropout` on
  an fp32 `[B·L, 1536]` tensor, q then v, layer by layer: the same kernel, shape, order and RNG
  offsets as peft's `nn.Dropout` in base Soup, so the masks are identical (GPU test
  `test_dropout_mask_is_the_one_torch_dropout_draws`). The kernel reads them through the
  packed→padded row map.
- **Same rounding as base.** Every value that base Soup rounds to fp16 is rounded to fp16 at the
  same point (projection outputs, LoRA outputs, SDPA inputs, MLP activation, dlogits incl. the
  GradScaler scale); everything base computes in fp32 we compute in fp32 (residual stream,
  norms, RoPE, softmax statistics). The only differences are summation order and fused GEMMs.
- **fp16 embedding storage** (`patches.py`): base upcasts the tied embedding to fp32; we keep
  the fp16 table (the upcast is exact) and return `.float()` of it from the embedding, so every
  downstream value is bit-identical. −467 MB resident.
- **Saved-activation budget.** Frozen 4-bit linears save nothing (no weight gradient), the MLP
  saves only fp16 gate\|up, norms save the fp32 input + `rstd`, LoRA saves its fp16 input.
- **Asynchronous offload** (`solution/offload.py`, our own code): those saved tensors of all but
  the last layer are copied to pinned host memory on a side CUDA stream during the forward
  pass and prefetched one layer ahead in backward, so the copy engine works in parallel with
  the SMs (idea from Unsloth's gradient offloading; implementation ours).
- **Metrics without logits** (`patches.py`): TRL's `entropy` / `mean_token_accuracy` /
  `num_tokens` are logged with TRL 0.29's own formulas from the head kernel's per-token
  results, and Soup's rewind recorder gets per-token losses through its documented
  `_soup_chunked_step` interface (the one Soup's own chunked-loss trainer uses).
- **Compile time.** Every run starts with an empty Triton cache, so `solution/warmup.py`
  compiles all kernel variants in a background thread (`JITFunction.warmup`, no GPU work) while
  Soup loads the model; the first fused step waits for it.

Everything falls back to the original code for anything it was not written for (eval, no
labels, left padding, other models, other LoRA targets, CPU), and `SOUP_KERNELS_DISABLE=1`
turns the solution off. Soup's 79 regression tests pass with the patches applied.

## 3. How to run

**Kaggle (what the jury runs):** open KAGGLE_NOTEBOOK_URL (made from
`templates/kaggle_notebook.ipynb`, `COMMIT` = FINAL_COMMIT_SHA), settings *GPU T4 x2*,
*Internet on*, then **Run All** (~10 min). It installs the pinned packages, clones this repo at
that commit, runs our kernel tests (`pytest solution`), `harness/run_task.py` (base, then ours),
`harness/check_rules.py` and `harness/score.py`.

**Locally on a Linux machine with a CUDA GPU** (only T4 numbers count):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q                    # our kernels vs PyTorch / bitsandbytes
python harness/run_task.py                      # base, then solution -> runs/result.json
python harness/check_rules.py                   # the 10 checks -> runs/checks.json
python harness/score.py runs/result.json
```

Without a GPU, the kernels run through Triton's interpreter:
`TRITON_INTERPRET=1 SOUP_KERNELS_TEST_CPU=1 python -m pytest solution -q`.

Switches (environment variables, all optional; defaults are what we submit):
`SOUP_KERNELS_DISABLE=1` (off), `SOUP_KERNELS_OFFLOAD_MLP=0`, `SOUP_KERNELS_OFFLOAD_RESID=0`,
`SOUP_KERNELS_OFFLOAD_KEEP_LAST=1`, `SOUP_KERNELS_HEAD_CHUNK=512`, `SOUP_KERNELS_GQA=1`,
`SOUP_KERNELS_WARMUP=1`.

## 4. Results

RESULTS_DETAIL

How we checked correctness before the GPU runs:

- `solution/test_kernels.py`: every kernel against PyTorch (forward and backward), NF4 decode
  bit-exact against `bitsandbytes.functional.dequantize_4bit`, dropout masks identical to
  `F.dropout`, offloaded activations bit-identical to kept ones.
- A tiny Qwen2 loaded in real NF4 with peft LoRA: fused forward/backward vs the HF/peft/bnb
  forward — loss within 4e-6 relative, per-row losses within 2e-5, LoRA gradients within 0.12 %.
- `soup train` end-to-end on that tiny model (Triton interpreter): `check_rules.compare_outputs`
  finds no difference to base (same logged metrics, tokens, rewind rows, adapter tensors), loss
  curve within 2.7e-6 (first step) / 2.3e-5 (mean).

## 5. Limitations and next steps

LIMITATIONS

## 6. Sources of ideas (all code here is ours)

- Fused linear cross-entropy computed chunk by chunk with the gradient written in the forward
  pass: Liger-Kernel (`FusedLinearCrossEntropy`), https://github.com/linkedin/Liger-Kernel;
  Soup's own `soup_cli/utils/stream_loss_chunk.py` for how to log TRL metrics without logits.
- Online softmax / single-pass logsumexp: M. Milakov, N. Gimelshein, *Online normalizer
  calculation for softmax*, 2018.
- NF4 and double quantization format: T. Dettmers et al., *QLoRA*, 2023, and the
  bitsandbytes reference implementation (`backends/default/ops.py`).
- Offloading saved activations to host memory during training: Unsloth's "smart gradient
  offloading", https://unsloth.ai/blog/long-context.
- Fused RoPE and fused RMSNorm kernels as a concept: Unsloth and Liger-Kernel.

## License

Apache-2.0, see `LICENSE` and `NOTICE` (starter kit and the five Soup test files). Data:
AmanMussa/kazakh-instruction-v2 (MIT); model: Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0).

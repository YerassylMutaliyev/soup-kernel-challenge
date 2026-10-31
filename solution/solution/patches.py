"""How the kernels plug into Soup. Every patch is narrow and falls back to the original code
whenever the model, the batch or the trainer is not the one it was written for (Soup's own
regression tests call these paths with fake modules).

1. peft.prepare_model_for_kbit_training: base Soup upcasts every fp16 non-4-bit parameter to
   fp32, including the 151,936 x 1,536 tied embedding / lm_head (933 MB in fp32). The upcast is
   exact (fp16 -> fp32), so we keep the embedding stored in fp16 (467 MB) and make it return
   exactly what the fp32 table returns (`.float()` of the same values); the lm_head GEMM runs
   in fp16 under autocast either way. Bit-identical outputs, -467 MB of resident memory.
2. Qwen2ForCausalLM.forward: a labelled training forward goes through fused_qwen2 (our
   kernels) and returns the loss without logits; anything else runs the original forward.
3. trl.SFTTrainer.compute_loss: TRL computes entropy / token accuracy from the logits and
   Soup's rewind recorder computes per-row losses from them. With the fused head those come
   from the head kernel instead and are logged by TRL's own formulas (the same way Soup's own
   chunked-loss trainer does it, soup_cli/utils/stream_loss_chunk.py); the rewind recorder gets
   the per-token losses through Soup's documented `_soup_chunked_step` attribute.
"""

import os

import torch

_PENDING = {}
_INSTALLED = False


def _enabled():
    return os.environ.get("SOUP_KERNELS_DISABLE", "") != "1"


def _on_device(t):
    # CUDA only; SOUP_KERNELS_ALLOW_CPU=1 lets our CPU integration test (Triton interpreter)
    # run the same path on a tiny model. It is never set in a measured run.
    return t.is_cuda or os.environ.get("SOUP_KERNELS_ALLOW_CPU") == "1"


# --------------------------------------------------------------------------- 1. fp16 embedding

def _tied_fp16_embedding(model):
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(model, Qwen2ForCausalLM):
        return None
    if not (getattr(model, "is_loaded_in_4bit", False)):
        return None
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if not isinstance(emb, torch.nn.Embedding) or head is None:
        return None
    w = emb.weight
    if w.dtype != torch.float16 or not _on_device(w) or head.weight is not w:
        return None
    if emb.max_norm is not None or emb.scale_grad_by_freq or emb.sparse:
        return None
    return emb


def _install_fp32_embedding_output(model, emb):
    head = model.get_output_embeddings()
    import torch.nn.functional as F

    def embedding_forward(input_ids):
        # Same values as the fp32 table base Soup builds from these fp16 weights.
        return F.embedding(input_ids, emb.weight, emb.padding_idx).float()

    def head_forward(hidden):
        w = head.weight
        if torch.is_autocast_enabled(hidden.device.type):
            return F.linear(hidden, w)  # autocast runs it in fp16, as with the fp32 table
        return F.linear(hidden.to(w.dtype), w).to(hidden.dtype)

    emb.forward = embedding_forward
    head.forward = head_forward


def _patch_prepare_kbit():
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_soup_kernels", False):
        return

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        emb = _tied_fp16_embedding(model) if _enabled() else None
        if emb is None:
            return original(model, *args, **kwargs)
        weight = emb.weight
        weight.requires_grad = False  # the original freezes it too; an int view needs it off
        weight.data = weight.data.view(torch.int16)  # not fp16/bf16: the fp32 upcast skips it
        try:
            out = original(model, *args, **kwargs)
        finally:
            weight.data = weight.data.view(torch.float16)
        weight.requires_grad = False
        _install_fp32_embedding_output(model, emb)
        model._soup_kernels_fp16_embedding = True
        return out

    prepare_model_for_kbit_training._soup_kernels = True
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    try:
        import peft.utils as peft_utils

        peft_utils.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- 2. model forward

def _causal_lm_of(model):
    seen = 0
    while seen < 6:
        seen += 1
        inner = getattr(model, "module", None)  # DDP / accelerate wrappers
        if isinstance(inner, torch.nn.Module):
            model = inner
            continue
        get_base = getattr(model, "get_base_model", None)
        if callable(get_base):
            base = get_base()
            if base is not model:
                model = base
                continue
        break
    return model


def _plan_for(lm):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if not isinstance(lm, Qwen2ForCausalLM):
        return None
    cached = lm.__dict__.get("_soup_kernels_plan")
    if cached is not None:
        return cached or None
    plan = False
    try:
        if _on_device(lm.model.embed_tokens.weight) and \
                lm.model.embed_tokens.weight.dtype == torch.float16:
            from .fused_qwen2 import ModelPlan

            plan = ModelPlan(lm)
            plan.gqa_sdpa = (lm.model.embed_tokens.weight.is_cuda and
                             _probe_gqa_sdpa(lm.model.embed_tokens.weight.device, plan))
    except (ValueError, AttributeError, TypeError, RuntimeError) as exc:
        print(f"[solution] fused path off for this model: {exc}", flush=True)
        plan = False
    lm.__dict__["_soup_kernels_plan"] = plan
    return plan or None


def _probe_gqa_sdpa(device, plan):
    from . import warmup

    lay = plan.layers[0]
    if warmup.GQA_SDPA is not None and (lay.n_heads, lay.n_kv, lay.head_dim) == (12, 2, 128):
        return warmup.GQA_SDPA  # already probed by the warm-up thread
    return warmup.probe_gqa_sdpa(lay.n_heads, lay.n_kv, lay.head_dim, device)


def _patch_causal_lm_forward():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    original = cls.forward
    if getattr(original, "_soup_kernels", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        pending = _PENDING
        if (pending.get("lm") is self and labels is not None and input_ids is not None
                and inputs_embeds is None and past_key_values is None and position_ids is None
                and self.training and torch.is_grad_enabled()
                and kwargs.get("shift_labels") is None and _on_device(input_ids)):
            from .fused_qwen2 import fused_loss_forward

            result = fused_loss_forward(self, pending["plan"], input_ids, attention_mask, labels,
                                        kwargs.get("num_items_in_batch"),
                                        pending["grad_scale"])
            if result is not None:
                loss, stats = result
                pending["stats"] = stats
                return CausalLMOutputWithPast(loss=loss, logits=None)
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._soup_kernels = True
    forward.__wrapped__ = original
    forward.__doc__ = original.__doc__
    cls.forward = forward


# --------------------------------------------------------------------------- 3. trainer

class _FusedStep:
    """What Soup's rewind recorder reads when a step has no logits (its #1693 interface)."""

    def __init__(self, stats):
        self.token_entropy = stats["token_entropy"]
        self.token_loss = stats["token_loss"]
        self.correct = stats["correct"]
        self.labelled = stats["labelled"]


def _grad_scale(trainer, device):
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    if scaler is None or not getattr(scaler, "is_enabled", lambda: False)():
        return torch.ones((), dtype=torch.float32, device=device)
    scale = getattr(scaler, "_scale", None)
    if scale is None:
        return torch.full((), float(getattr(scaler, "_init_scale", 2.0 ** 16)),
                          dtype=torch.float32, device=device)
    return scale.detach()


def _log_metrics(trainer, mode, inputs, stats):
    """entropy, num_tokens and mean_token_accuracy exactly as TRL 0.29 logs them."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    entropy = torch.sum(stats["token_entropy"] * attention_mask) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct = gather(stats["correct"])
    total = gather(stats["labelled"])
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _log_metrics_from_logits(trainer, mode, inputs, outputs):
    """TRL 0.29's SFTTrainer.compute_loss metric block, for a batch that took the original
    forward (it has logits)."""
    from trl.trainer.utils import entropy_from_logits

    gather = trainer.accelerator.gather_for_metrics
    with torch.no_grad():
        per_token_entropy = entropy_from_logits(outputs.logits)
        attention_mask = inputs["attention_mask"]
        entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    with torch.no_grad():
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct = gather(((predictions == shift_labels) & mask).sum())
        total = gather(mask.sum())
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _patch_sft_compute_loss():
    from transformers import Trainer
    from trl import SFTTrainer

    original = SFTTrainer.compute_loss
    if getattr(original, "_soup_kernels", False):
        return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        plan = lm = None
        try:
            if (_enabled() and isinstance(model, torch.nn.Module) and model.training
                    and isinstance(inputs, dict) and "labels" in inputs
                    and "attention_mask" in inputs and "shift_labels" not in inputs
                    and not getattr(self.args, "use_liger_kernel", False)
                    and getattr(self, "num_virtual_tokens", 0) == 0
                    and not getattr(self, "aux_loss_enabled", False)
                    and getattr(self, "compute_loss_func", None) is None
                    and getattr(self, "label_smoother", None) is None
                    and getattr(self.args, "loss_type", "nll") in ("nll", None)
                    and hasattr(self, "_metrics")):
                lm = _causal_lm_of(model)
                plan = _plan_for(lm)
        except Exception as exc:  # noqa: BLE001 - never break training on a probe
            print(f"[solution] fused path off: {exc}", flush=True)
            plan = None
        if plan is None:
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)

        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        device = inputs["input_ids"].device
        _PENDING.clear()
        _PENDING.update(lm=lm, plan=plan, grad_scale=_grad_scale(self, device), stats=None)
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            stats = _PENDING.get("stats")
        finally:
            _PENDING.clear()
        if stats is None:  # the batch was not one the fused path takes: TRL's own metrics
            _log_metrics_from_logits(self, mode, inputs, outputs)
            setattr(self, "_soup_chunked_step", None)
        else:
            _log_metrics(self, mode, inputs, stats)
            setattr(self, "_soup_chunked_step", _FusedStep(stats))
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_kernels = True
    compute_loss.__wrapped__ = original
    SFTTrainer.compute_loss = compute_loss


def install():
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return
    _patch_prepare_kbit()
    _patch_causal_lm_forward()
    _patch_sft_compute_loss()
    _INSTALLED = True

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Runtime:
    model_name: str
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    layers: Any
    final_norm: Any
    lm_head: Any
    attn_implementation: str


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def decoder_layers(model: Any):
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, child_name, None) if parent is not None else None
        if layers is not None:
            return layers
    raise ValueError("Could not locate decoder layers for this architecture.")


def final_norm(model: Any):
    candidates = (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    )
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        norm = getattr(parent, child_name, None) if parent is not None else None
        if norm is not None:
            return norm
    raise ValueError("Could not locate the final normalization layer.")


def load_runtime(
    model_name: str,
    trust_remote_code: bool = False,
    attn_implementation: str = "eager",
) -> Runtime:
    """Load a causal LM and its tokenizer.

    ``attn_implementation`` defaults to ``"eager"`` because the workbench
    needs attention weights for its Attention view; the optimized kernels
    (SDPA / flash) run faster but do not expose the attention matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    return Runtime(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
        layers=decoder_layers(model),
        final_norm=final_norm(model),
        lm_head=model.get_output_embeddings(),
        attn_implementation=attn_implementation,
    )


def unload_runtime(runtime: Runtime | None = None) -> None:
    if runtime is not None:
        runtime.model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize(runtime: Runtime, prompt: str, max_length: int = 256):
    return runtime.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).to(runtime.device)


def token_labels(runtime: Runtime, input_ids: torch.Tensor) -> list[str]:
    labels = []
    for token_id in input_ids[0].detach().cpu().tolist():
        text = runtime.tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        labels.append(f"{len(labels)}: {text!r}")
    return labels


def layer_modules(layer: Any) -> tuple[Any, Any]:
    """Return (attn_module, mlp_module) for a decoder layer, or (None, None).

    Follows the common per-layer submodule convention used by Llama, Qwen,
    and Mistral models: ``layer.self_attn`` and ``layer.mlp``. Architectures
    that do not follow this convention (e.g. GPT-2, which nests attention
    and MLP inside a single ``layer.h`` block) return ``(None, None)`` so
    that the caller can fall back to residual-stream patching only.
    """
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        mlp = getattr(layer, "mlp_module", None)
    if attn is None and mlp is None:
        return (None, None)
    return (attn, mlp)


def cuda_stats() -> dict[str, str]:
    if not torch.cuda.is_available():
        return {"Device": "CPU"}
    return {
        "Device": torch.cuda.get_device_name(0),
        "Allocated": f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB",
        "Reserved": f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB",
    }


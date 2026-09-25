"""Native MLP neuron, Q/K/V, and path-level experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .analysis import top_tokens
from .components import component_capabilities
from .runtime import Runtime, tokenize


def _mlp(runtime: Runtime, layer: int) -> Any:
    module = getattr(runtime.layers[layer], "mlp", None)
    if module is None:
        module = getattr(runtime.layers[layer], "mlp_module", None)
    if module is None or getattr(module, "down_proj", None) is None:
        raise ValueError("This architecture does not expose a separable MLP intermediate.")
    return module


def _attention(runtime: Runtime, layer: int) -> Any:
    module = getattr(runtime.layers[layer], "self_attn", None)
    if module is None:
        module = getattr(runtime.layers[layer], "attention", None)
    if module is None or not all(getattr(module, name, None) is not None for name in ("q_proj", "k_proj", "v_proj")):
        raise ValueError("This architecture does not expose separable Q/K/V projections.")
    return module


def mlp_neuron_activations(runtime: Runtime, prompt: str, layer: int,
                           max_length: int = 128) -> pd.DataFrame:
    """Capture the per-token intermediate vector entering an MLP down projection."""
    mlp = _mlp(runtime, layer)
    batch = tokenize(runtime, prompt, max_length=max_length)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        if not inputs:
            raise ValueError("MLP down projection did not expose an intermediate input.")
        captured["value"] = inputs[0].detach().float().cpu()

    handle = mlp.down_proj.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            runtime.model(**batch, use_cache=False)
    finally:
        handle.remove()
    values = captured["value"][0]
    rows = []
    for position in range(values.shape[0]):
        for neuron in range(values.shape[1]):
            rows.append({
                "layer": layer,
                "position": position,
                "neuron": neuron,
                "activation": float(values[position, neuron]),
                "magnitude": float(values[position, neuron].abs()),
            })
    return pd.DataFrame(rows)


def top_mlp_neurons(activations: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Return the strongest neurons by absolute activation."""
    if activations.empty:
        return activations.copy()
    return activations.nlargest(k, "magnitude").reset_index(drop=True)


def _replace_neuron_input(inputs: tuple[Any, ...], position: int, neuron: int) -> tuple[Any, ...]:
    if not inputs:
        raise ValueError("MLP projection did not expose an input tensor.")
    value = inputs[0].clone()
    if value.ndim != 3 or position < 0 or position >= value.shape[1] or neuron < 0 or neuron >= value.shape[2]:
        raise ValueError("The selected token position or MLP neuron is outside the available tensor.")
    value[:, position, neuron] = 0
    return (value, *inputs[1:])


def ablate_mlp_neuron(runtime: Runtime, prompt: str, layer: int, neuron: int,
                      position: int = -1, target_token_id: int | None = None,
                      max_length: int = 128, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare one MLP neuron against a single-token zero ablation."""
    mlp = _mlp(runtime, layer)
    batch = tokenize(runtime, prompt, max_length=max_length)
    sequence_length = int(batch["input_ids"].shape[1])
    position = sequence_length + position if position < 0 else position
    with torch.inference_mode():
        baseline_logits = runtime.model(**batch, use_cache=False).logits[0, -1].float()
    if target_token_id is None:
        target_token_id = int(baseline_logits.argmax())

    def ablate(_module, inputs):
        return _replace_neuron_input(inputs, position, neuron)

    handle = mlp.down_proj.register_forward_pre_hook(ablate)
    try:
        with torch.inference_mode():
            ablated_logits = runtime.model(**batch, use_cache=False).logits[0, -1].float()
    finally:
        handle.remove()
    metadata = {
        "layer": layer, "neuron": neuron, "position": position,
        "target_token_id": target_token_id,
        "baseline_logit": float(baseline_logits[target_token_id]),
        "ablated_logit": float(ablated_logits[target_token_id]),
        "delta_logit": float(ablated_logits[target_token_id] - baseline_logits[target_token_id]),
    }
    return top_tokens(runtime, baseline_logits.cpu(), k), top_tokens(runtime, ablated_logits.cpu(), k), metadata


def attention_signal_summary(runtime: Runtime, prompt: str, layer: int, signal: str,
                            head: int | None = None, max_length: int = 128) -> pd.DataFrame:
    """Summarise Q, K, or V vector magnitudes by token position and head."""
    if signal not in {"q", "k", "v"}:
        raise ValueError("Signal must be one of 'q', 'k', or 'v'.")
    attention = _attention(runtime, layer)
    projection = getattr(attention, f"{signal}_proj")
    batch = tokenize(runtime, prompt, max_length=max_length)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["value"] = output.detach().float().cpu()

    handle = projection.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            runtime.model(**batch, use_cache=False)
    finally:
        handle.remove()
    values = captured["value"][0]
    config = runtime.model.config
    if signal == "q":
        heads = int(getattr(config, "num_attention_heads", 0))
    else:
        heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 0)))
    if not heads or values.shape[-1] % heads:
        raise ValueError("The projection output cannot be split into attention heads.")
    head_size = values.shape[-1] // heads
    values = values.reshape(values.shape[0], heads, head_size)
    selected_heads = range(heads) if head is None else [head]
    if head is not None and (head < 0 or head >= heads):
        raise ValueError(f"Head {head} is outside the available {signal.upper()} head range.")
    rows = []
    for position in range(values.shape[0]):
        for selected in selected_heads:
            vector = values[position, selected]
            rows.append({
                "layer": layer, "signal": signal.upper(), "position": position,
                "head": selected, "magnitude": float(vector.norm()),
                "mean": float(vector.mean()), "max_abs": float(vector.abs().max()),
            })
    return pd.DataFrame(rows)


def patch_mlp_neuron(runtime: Runtime, source_prompt: str, target_prompt: str,
                     layer: int, neuron: int, position: int = -1,
                     target_token_id: int | None = None, max_length: int = 128,
                     k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Copy one source MLP neuron into the target at the selected layer/position."""
    mlp = _mlp(runtime, layer)
    source_batch = tokenize(runtime, source_prompt, max_length=max_length)
    target_batch = tokenize(runtime, target_prompt, max_length=max_length)
    source_position = source_batch["input_ids"].shape[1] + position if position < 0 else position
    target_position = target_batch["input_ids"].shape[1] + position if position < 0 else position
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        captured["value"] = inputs[0][:, source_position, neuron].detach().clone()

    handle = mlp.down_proj.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            runtime.model(**source_batch, use_cache=False)
    finally:
        handle.remove()
    with torch.inference_mode():
        baseline_logits = runtime.model(**target_batch, use_cache=False).logits[0, -1].float()
    if target_token_id is None:
        target_token_id = int(baseline_logits.argmax())

    def inject(_module, inputs):
        if not inputs:
            raise ValueError("MLP down projection did not expose an input tensor.")
        value = inputs[0].clone()
        value[:, target_position, neuron] = captured["value"].to(value.device, value.dtype)
        return (value, *inputs[1:])

    handle = mlp.down_proj.register_forward_pre_hook(inject)
    try:
        with torch.inference_mode():
            patched_logits = runtime.model(**target_batch, use_cache=False).logits[0, -1].float()
    finally:
        handle.remove()
    metadata = {
        "source_prompt": source_prompt, "target_prompt": target_prompt,
        "layer": layer, "neuron": neuron, "position": target_position,
        "target_token_id": target_token_id,
        "delta_logit": float(patched_logits[target_token_id] - baseline_logits[target_token_id]),
    }
    return top_tokens(runtime, baseline_logits.cpu(), k), top_tokens(runtime, patched_logits.cpu(), k), metadata

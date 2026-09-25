"""Architecture-aware helpers for fine-grained component inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComponentCapabilities:
    """Supported inspectable components for one decoder layer."""

    mlp_neurons: bool
    attention_qkv: bool
    attention_head_count: int
    reason: str = ""


def component_capabilities(runtime: Any, layer: int) -> ComponentCapabilities:
    """Detect separable projections without importing provider-specific code."""
    if layer < 0 or layer >= len(runtime.layers):
        raise ValueError("Layer is outside the available range.")
    decoder_layer = runtime.layers[layer]
    attention = getattr(decoder_layer, "self_attn", None) or getattr(decoder_layer, "attention", None)
    mlp = getattr(decoder_layer, "mlp", None) or getattr(decoder_layer, "mlp_module", None)
    reasons = []
    mlp_neurons = bool(mlp is not None and getattr(mlp, "down_proj", None) is not None)
    if not mlp_neurons:
        reasons.append("the MLP does not expose a separable down projection")
    qkv = bool(
        attention is not None
        and all(getattr(attention, name, None) is not None for name in ("q_proj", "k_proj", "v_proj"))
    )
    if not qkv:
        reasons.append("attention Q/K/V projections are not exposed")
    heads = int(getattr(getattr(runtime.model, "config", None), "num_attention_heads", 0) or 0)
    return ComponentCapabilities(mlp_neurons, qkv, heads, "; ".join(reasons))


def component_cache_key(
    model_name: str,
    prompt_or_data: str,
    layer: int,
    target: str,
    positions: tuple[int, ...] = (),
    head: int | None = None,
    neuron: int | None = None,
    intervention: str | None = None,
    artefact: str | None = None,
) -> tuple[object, ...]:
    """Build an identity for reusable fine-grained results."""
    return (
        model_name, prompt_or_data, int(layer), target, positions,
        head, neuron, intervention, artefact,
    )

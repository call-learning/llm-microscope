"""Lazy adapters for optional interpretability providers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch


@dataclass(frozen=True)
class AdapterStatus:
    provider: str
    package: str
    installed: bool
    compatible: bool | None
    artefact_ready: bool | None
    adapter_ready: bool
    render_ready: bool | None
    message: str
    install_hint: str


@dataclass(frozen=True)
class AdapterResult:
    provider: str
    kind: str
    value: Any
    labels: dict[str, Any]
    assumptions: str


PROVIDERS = {
    "BertViz": ("bertviz", "pip install bertviz"),
    "CircuitsVis": ("circuitsvis", "pip install circuitsvis"),
    "Tuned Lens": ("tuned_lens", "pip install tuned-lens"),
    "TransformerLens": ("transformer_lens", "uv sync --extra transformer-lens"),
    "NNsight": ("nnsight", "uv sync --extra nnsight"),
    "SAELens": ("sae_lens", "uv sync --extra saelens"),
    "Pyvene": ("pyvene", "uv sync --extra pyvene"),
}


def provider_status(provider: str) -> AdapterStatus:
    """Return status without importing the optional provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    package, install_hint = PROVIDERS[provider]
    installed = find_spec(package) is not None
    adapter_ready = installed
    return AdapterStatus(
        provider=provider,
        package=package,
        installed=installed,
        compatible=None,
        artefact_ready=None,
        adapter_ready=adapter_ready,
        render_ready=None,
        message=(
            f"{provider} is ready for an explicit compatibility check."
            if installed else f"{provider} is not installed."
        ),
        install_hint=install_hint,
    )


def _unavailable(provider: str, detail: str) -> RuntimeError:
    return RuntimeError(f"{provider} adapter unavailable: {detail}. Native views remain available.")


def bertviz_attention(attentions: tuple[torch.Tensor, ...], labels: list[str],
                      layer: int | None = None, heads: list[int] | None = None) -> AdapterResult:
    """Prepare BertViz HTML from native attention tensors."""
    status = provider_status("BertViz")
    if not status.installed:
        raise _unavailable("BertViz", status.install_hint)
    if not attentions:
        raise _unavailable("BertViz", "attention matrices were not captured")
    try:
        from bertviz import head_view

        matrices = [attention.float().cpu() for attention in attentions]
        selected = matrices if layer is None else matrices[layer:layer + 1]
        html = head_view(
            attention=selected,
            tokens=[label.split(": ", 1)[-1].strip("'") for label in labels],
            heads=heads,
            html_action="return",
        )
        html_text = html if isinstance(html, str) else getattr(html, "data", str(html))
        return AdapterResult("BertViz", "html", html_text,
                             {"layer": layer, "heads": heads, "tokens": labels},
                             "Descriptive attention weights; not causal evidence.")
    except Exception as exc:
        raise _unavailable("BertViz", f"rendering failed: {type(exc).__name__}: {exc}") from exc


def circuitsvis_attention(attention: torch.Tensor, labels: list[str],
                          heads: list[int] | None = None) -> AdapterResult:
    """Prepare CircuitsVis embeddable HTML for one layer of attention."""
    status = provider_status("CircuitsVis")
    if not status.installed:
        raise _unavailable("CircuitsVis", status.install_hint)
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise _unavailable("CircuitsVis", "expected attention shape (1, heads, query, key)")
    try:
        import circuitsvis

        selected = attention[0].float().cpu()
        if heads is not None:
            selected = selected[heads]
        rendered = circuitsvis.attention.attention_heads(
            attention=selected,
            tokens=[label.split(": ", 1)[-1].strip("'") for label in labels],
            attention_head_names=[f"Head {index}" for index in (heads or range(selected.shape[0]))],
        )
        return AdapterResult("CircuitsVis", "html", rendered._repr_html_(),
                             {"heads": heads, "tokens": labels},
                             "Interactive attention rendering; not causal evidence.")
    except Exception as exc:
        raise _unavailable("CircuitsVis", f"rendering failed: {type(exc).__name__}: {exc}") from exc


def transformer_lens_model(model_name: str, device: str = "cpu") -> Any:
    status = provider_status("TransformerLens")
    if not status.installed:
        raise _unavailable("TransformerLens", status.install_hint)
    try:
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained_no_processing(model_name, device=device)
    except Exception as exc:
        raise _unavailable("TransformerLens", f"model loading failed: {type(exc).__name__}: {exc}") from exc


def nnsight_model(model_name: str, device_map: str | None = None) -> Any:
    status = provider_status("NNsight")
    if not status.installed:
        raise _unavailable("NNsight", status.install_hint)
    try:
        from nnsight import LanguageModel

        kwargs = {} if device_map is None else {"device_map": device_map}
        return LanguageModel(model_name, **kwargs)
    except Exception as exc:
        raise _unavailable("NNsight", f"model loading failed: {type(exc).__name__}: {exc}") from exc


def pyvene_status() -> AdapterStatus:
    """Report Pyvene availability; model layout validation is deferred to use."""
    return provider_status("Pyvene")


def pyvene_prepare(model: Any, intervention_config: Any) -> Any:
    """Create a Pyvene intervenable model only on explicit request."""
    status = pyvene_status()
    if not status.installed:
        raise _unavailable("Pyvene", status.install_hint)
    try:
        from pyvene import IntervenableConfig, IntervenableModel

        config = intervention_config
        if not isinstance(config, IntervenableConfig):
            config = IntervenableConfig(config)
        return IntervenableModel(config, model)
    except Exception as exc:
        raise _unavailable("Pyvene", f"intervention setup failed: {type(exc).__name__}: {exc}") from exc


def saelens_load(release: str, sae_id: str, device: str = "cpu") -> Any:
    """Load a SAELens SAE only when an exact release and SAE id are supplied."""
    status = provider_status("SAELens")
    if not status.installed:
        raise _unavailable("SAELens", status.install_hint)
    try:
        from sae_lens import SAE

        return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    except Exception as exc:
        raise _unavailable("SAELens", f"artefact loading failed: {type(exc).__name__}: {exc}") from exc

"""Optional tuned-lens boundary.

The core workbench deliberately does not import this provider at startup. A
tuned lens is only valid when its translator was trained for the selected
checkpoint and hidden-state representation.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from dataclasses import dataclass
from typing import Any

import torch


def provider_status() -> dict[str, str | bool]:
    """Report availability without importing the provider module."""
    available = find_spec("tuned_lens") is not None
    return {
        "installed": available,
        "adapter_enabled": True,
        "message": (
            "tuned_lens is installed; provide a validated translator artefact for the native adapter."
            if available else
            "Install the optional tuned-lens provider to evaluate tuned decoding."
        ),
    }


def load_translator(model_name: str, artefact_path: str | None = None) -> Any:
    """Load a provider translator only when explicitly requested.

    This intentionally raises an actionable error until a verified provider
    adapter and checkpoint artefact contract are selected for the model.
    """
    status = provider_status()
    if not status["installed"]:
        raise RuntimeError("Tuned lens requires the optional 'tuned-lens' provider.")
    if not artefact_path:
        raise RuntimeError(
            f"No tuned-lens translator artefact was supplied for {model_name!r}; "
            "an exact model/checkpoint match is required."
        )
    provider = importlib.import_module("tuned_lens")
    loader = getattr(provider, "load_model", None)
    if loader is None:
        raise RuntimeError("Installed tuned-lens provider has no supported model loader.")
    return loader(artefact_path, model_name=model_name)


@dataclass(frozen=True)
class TunedLensArtifact:
    """Portable affine translator artefact used by the native adapter.

    The file is a torch checkpoint containing ``model_name``, ``hidden_size``,
    ``weights`` and ``biases`` lists. It is deliberately explicit so an
    artefact trained for another checkpoint cannot be silently reused.
    """

    model_name: str
    hidden_size: int
    weights: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor, ...]


def load_artifact(path: str, model_name: str, hidden_size: int) -> TunedLensArtifact:
    """Load and validate a native tuned-lens checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Tuned-lens artefact must be a checkpoint dictionary.")
    if payload.get("model_name") != model_name:
        raise ValueError("Tuned-lens artefact model_name does not match the loaded model.")
    if int(payload.get("hidden_size", -1)) != hidden_size:
        raise ValueError("Tuned-lens artefact hidden_size does not match the loaded model.")
    weights = tuple(torch.as_tensor(weight).float() for weight in payload.get("weights", ()))
    biases = tuple(torch.as_tensor(bias).float() for bias in payload.get("biases", ()))
    if not weights or len(weights) != len(biases):
        raise ValueError("Tuned-lens artefact must contain equally sized weights and biases lists.")
    if any(weight.shape != (hidden_size, hidden_size) for weight in weights):
        raise ValueError("Every tuned-lens weight must have shape (hidden_size, hidden_size).")
    if any(bias.shape != (hidden_size,) for bias in biases):
        raise ValueError("Every tuned-lens bias must have shape (hidden_size,).")
    return TunedLensArtifact(model_name, hidden_size, weights, biases)


def decode_artifact(runtime: Any, hidden_states: tuple[torch.Tensor, ...],
                    artifact: TunedLensArtifact, token_id: int,
                    reference_logits: torch.Tensor | None = None) -> pd.DataFrame:
    """Decode one selected token through a validated native tuned lens."""
    import pandas as pd

    if len(artifact.weights) != len(hidden_states) - 1:
        raise ValueError("Tuned-lens artefact layer count does not match hidden states.")
    rows = []
    for layer, (hidden, weight, bias) in enumerate(zip(hidden_states[1:], artifact.weights, artifact.biases)):
        state = hidden[0, -1].float()
        translated = state @ weight.T + bias
        with torch.inference_mode():
            logits = runtime.lm_head(runtime.final_norm(translated.to(runtime.device, runtime.dtype))).float().cpu()
        probabilities = logits.softmax(dim=-1)
        rank = int(1 + (logits > logits[token_id]).sum())
        kl_to_final = float("nan")
        if reference_logits is not None:
            reference = reference_logits.float().softmax(dim=-1)
            kl_to_final = float((probabilities * (
                probabilities.clamp_min(1e-12).log() - reference.clamp_min(1e-12).log()
            )).sum())
        rows.append({
            "layer": layer,
            "logit": float(logits[token_id]),
            "probability": float(probabilities[token_id]),
            "rank": rank,
            "token_id": token_id,
            "kl_to_final": kl_to_final,
        })
    return pd.DataFrame(rows)

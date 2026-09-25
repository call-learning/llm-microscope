"""Optional, explicit sparse-autoencoder artefact support."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch


@dataclass(frozen=True)
class SAEArtifact:
    model_name: str
    layer: int
    hidden_size: int
    encoder: torch.Tensor
    decoder: torch.Tensor
    threshold: float = 0.0

    @property
    def feature_count(self) -> int:
        return int(self.encoder.shape[0])


def provider_status() -> dict[str, str | bool]:
    installed = find_spec("sae_lens") is not None
    return {
        "installed": installed,
        "adapter_enabled": True,
        "message": (
            "SAELens is installed; provide an exact compatible SAE artefact."
            if installed else
            "SAELens is not installed. Native component views remain available."
        ),
    }


def load_artifact(path: str, model_name: str, layer: int, hidden_size: int) -> SAEArtifact:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("SAE artefact must be a checkpoint dictionary.")
    if payload.get("model_name") != model_name:
        raise ValueError("SAE model_name does not match the loaded model.")
    if int(payload.get("layer", -1)) != layer:
        raise ValueError("SAE layer does not match the selected model layer.")
    if int(payload.get("hidden_size", -1)) != hidden_size:
        raise ValueError("SAE hidden_size does not match the model representation.")
    encoder = torch.as_tensor(payload.get("encoder", ())).float()
    decoder = torch.as_tensor(payload.get("decoder", ())).float()
    if encoder.ndim != 2 or decoder.shape != (hidden_size, encoder.shape[0]):
        raise ValueError("SAE encoder/decoder shapes are incompatible with the model hidden size.")
    if encoder.shape[1] != hidden_size:
        raise ValueError("SAE encoder input size does not match the model hidden size.")
    return SAEArtifact(model_name, layer, hidden_size, encoder, decoder, float(payload.get("threshold", 0.0)))


def encode(activations: torch.Tensor, artifact: SAEArtifact) -> torch.Tensor:
    """Encode residual activations as non-negative sparse feature values."""
    if activations.shape[-1] != artifact.hidden_size:
        raise ValueError("Activation hidden size does not match the SAE artefact.")
    features = activations.float() @ artifact.encoder.T
    return (features - artifact.threshold).relu()


def ablate_feature(activation: torch.Tensor, artifact: SAEArtifact, feature: int) -> torch.Tensor:
    if feature < 0 or feature >= artifact.feature_count:
        raise ValueError("SAE feature is outside the available range.")
    features = encode(activation, artifact)
    features[..., feature] = 0
    return features @ artifact.decoder.T

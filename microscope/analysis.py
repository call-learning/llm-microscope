from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .runtime import Runtime, token_labels, tokenize


@dataclass
class AnalysisResult:
    input_ids: torch.Tensor
    labels: list[str]
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attentions: tuple[torch.Tensor, ...] | None


def analysis_cache_key(
    model_name: str,
    prompt_or_data: str,
    max_length: int,
    method: str,
    target_token_id: int | None = None,
    layer_range: tuple[int, int] | None = None,
    artefact: str | None = None,
) -> tuple[object, ...]:
    """Build a stable identity for derived analysis data."""
    return (
        model_name,
        prompt_or_data,
        int(max_length),
        method,
        target_token_id,
        layer_range,
        artefact,
    )


def analyse(
    runtime: Runtime,
    prompt: str,
    max_length: int = 256,
    with_attention: bool = True,
) -> AnalysisResult:
    batch = tokenize(runtime, prompt, max_length=max_length)
    with torch.inference_mode():
        outputs = runtime.model(
            **batch,
            output_hidden_states=True,
            output_attentions=with_attention,
            use_cache=False,
            return_dict=True,
        )
    hidden = tuple(t.detach().float().cpu() for t in outputs.hidden_states)
    attentions = (
        tuple(t.detach().float().cpu() for t in outputs.attentions)
        if outputs.attentions is not None
        else ()
    )
    return AnalysisResult(
        input_ids=batch["input_ids"].detach().cpu(),
        labels=token_labels(runtime, batch["input_ids"]),
        logits=outputs.logits.detach().float().cpu(),
        hidden_states=hidden,
        attentions=attentions,
    )


def decode_token(runtime: Runtime, token_id: int) -> str:
    return runtime.tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=False,
    )


def top_tokens(runtime: Runtime, logits: torch.Tensor, k: int = 10) -> pd.DataFrame:
    probs = logits.float().softmax(dim=-1)
    values, indices = probs.topk(min(k, probs.shape[-1]))
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "token": repr(decode_token(runtime, int(token_id))),
                "token_id": int(token_id),
                "probability": float(prob),
            }
            for rank, (prob, token_id) in enumerate(zip(values, indices), 1)
        ]
    )


def project_hidden(runtime: Runtime, hidden: torch.Tensor) -> torch.Tensor:
    state = hidden.to(runtime.device, dtype=runtime.dtype)
    with torch.inference_mode():
        return runtime.lm_head(runtime.final_norm(state)).float().cpu()


def logit_lens(runtime: Runtime, result: AnalysisResult, k: int = 5) -> pd.DataFrame:
    rows = []
    # hidden_states[0] is the embedding output; subsequent entries are layers.
    for layer, hidden in enumerate(result.hidden_states[1:]):
        logits = project_hidden(runtime, hidden[:, -1, :])[0]
        top = top_tokens(runtime, logits, k=k)
        row: dict[str, object] = {"layer": layer}
        for item in top.itertuples(index=False):
            row[f"#{item.rank} token"] = item.token
            row[f"#{item.rank} probability"] = item.probability
        rows.append(row)
    return pd.DataFrame(rows)


def final_token_evolution(runtime: Runtime, result: AnalysisResult) -> pd.DataFrame:
    final_id = int(result.logits[0, -1].argmax())
    rows = []
    for layer, hidden in enumerate(result.hidden_states[1:]):
        logits = project_hidden(runtime, hidden[:, -1, :])[0]
        probability = float(logits.softmax(dim=-1)[final_id])
        rows.append({"layer": layer, "probability": probability})
    df = pd.DataFrame(rows)
    df.attrs["token"] = repr(decode_token(runtime, final_id))
    return df


def layer_prediction_metrics(
    runtime: Runtime,
    result: AnalysisResult,
    token_id: int,
) -> pd.DataFrame:
    """Return selected-token prediction metrics for every layer and position.

    The hidden states are projected one layer at a time and only scalar values
    for ``token_id`` are retained. This avoids keeping a full vocabulary-sized
    tensor for all layers in memory.
    """
    if token_id < 0 or token_id >= runtime.lm_head.out_features:
        raise ValueError(f"Token ID must be between 0 and {runtime.lm_head.out_features - 1}.")

    rows: list[dict[str, object]] = []
    for layer, hidden in enumerate(result.hidden_states[1:]):
        logits = project_hidden(runtime, hidden[0])
        probabilities = logits.float().softmax(dim=-1)
        reference = result.logits[0].float().softmax(dim=-1)
        kl_to_final = (probabilities * (
            probabilities.clamp_min(1e-12).log() - reference.clamp_min(1e-12).log()
        )).sum(dim=-1)
        selected_logits = logits[:, token_id]
        selected_probabilities = probabilities[:, token_id]
        ranks = 1 + (logits > selected_logits.unsqueeze(-1)).sum(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        for position, (logit, probability, rank, position_entropy) in enumerate(
            zip(selected_logits, selected_probabilities, ranks, entropy)
        ):
            rows.append(
                {
                    "layer": layer,
                    "position": position,
                    "token": result.labels[position],
                    "token_id": token_id,
                    "logit": float(logit),
                    "probability": float(probability),
                    "rank": int(rank),
                    "entropy": float(position_entropy),
                    "kl_to_final": float(kl_to_final[position]),
                }
            )
    return pd.DataFrame(rows)


def layer_norms(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": list(range(len(result.hidden_states) - 1)),
            "norm": [float(h[0, -1].norm()) for h in result.hidden_states[1:]],
        }
    )


def representation_metrics(result: AnalysisResult, metric: str = "norm") -> pd.DataFrame:
    """Return token-by-layer scalar representation metrics.

    ``norm`` measures hidden-state magnitude. ``cosine_change`` measures the
    change from the preceding residual state at the same token position.
    """
    if metric not in {"norm", "cosine_change"}:
        raise ValueError("Metric must be 'norm' or 'cosine_change'.")

    rows: list[dict[str, object]] = []
    for layer, hidden in enumerate(result.hidden_states[1:]):
        state = hidden[0]
        if metric == "norm":
            values = state.norm(dim=-1)
        else:
            previous = result.hidden_states[layer][0]
            values = 1 - F.cosine_similarity(state, previous, dim=-1)
        for position, value in enumerate(values):
            rows.append(
                {
                    "layer": layer,
                    "position": position,
                    "token": result.labels[position],
                    "value": float(value),
                    "metric": metric,
                }
            )
    return pd.DataFrame(rows)


def pairwise_cosine_distances(result: AnalysisResult, layer: int) -> pd.DataFrame:
    """Return a token-by-token cosine distance matrix for one layer."""
    if layer < 0 or layer >= len(result.hidden_states) - 1:
        raise ValueError("Layer is outside the available hidden-state range.")
    state = result.hidden_states[layer + 1][0]
    normalized = F.normalize(state, dim=-1)
    distance = 1 - normalized @ normalized.T
    return pd.DataFrame(
        distance.numpy(),
        index=result.labels,
        columns=result.labels,
    )


def attention_head_summary(
    attentions: tuple[torch.Tensor, ...] | None,
    labels: list[str],
) -> pd.DataFrame:
    """Summarise concentration and maximum destination for every head/query."""
    if not attentions:
        return pd.DataFrame(
            columns=[
                "layer", "head", "query_position", "query_token",
                "entropy", "max_key_position", "max_key_token", "max_attention",
            ]
        )

    rows: list[dict[str, object]] = []
    for layer, layer_attention in enumerate(attentions):
        if layer_attention.ndim != 4 or layer_attention.shape[0] != 1:
            raise ValueError("Attention tensors must have shape (1, heads, query, key).")
        values = layer_attention[0].float()
        if values.shape[-1] != len(labels) or values.shape[-2] != len(labels):
            raise ValueError("Attention dimensions must match the token labels.")
        entropy = -(values * values.clamp_min(1e-12).log()).sum(dim=-1)
        maximum, positions = values.max(dim=-1)
        for head in range(values.shape[0]):
            for query_position in range(values.shape[1]):
                key_position = int(positions[head, query_position])
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_position": query_position,
                        "query_token": labels[query_position],
                        "entropy": float(entropy[head, query_position]),
                        "max_key_position": key_position,
                        "max_key_token": labels[key_position],
                        "max_attention": float(maximum[head, query_position]),
                    }
                )
    return pd.DataFrame(rows)


def attention_rollout(
    attentions: tuple[torch.Tensor, ...] | None,
    include_residual: bool = True,
) -> torch.Tensor:
    """Aggregate attention across heads and layers with documented assumptions.

    Each layer is averaged across heads. When ``include_residual`` is true, an
    identity matrix is added before row normalization. The normalized matrices
    are composed from the earliest layer to the latest layer.
    """
    if not attentions:
        raise ValueError("Attention rollout requires captured attention matrices.")
    first = attentions[0]
    if first.ndim != 4 or first.shape[0] != 1 or first.shape[-1] != first.shape[-2]:
        raise ValueError("Attention rollout requires square (1, heads, seq, seq) tensors.")
    sequence_length = first.shape[-1]
    joint = torch.eye(sequence_length, dtype=torch.float32)
    for layer_attention in attentions:
        if (
            layer_attention.ndim != 4
            or layer_attention.shape[0] != 1
            or layer_attention.shape[-2:] != (sequence_length, sequence_length)
        ):
            raise ValueError("All attention tensors must share square sequence dimensions.")
        matrix = layer_attention[0].float().mean(dim=0)
        if include_residual:
            matrix = matrix + torch.eye(sequence_length, dtype=matrix.dtype)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = matrix @ joint
    return joint


def attention_head_similarity(attentions: tuple[torch.Tensor, ...], layer: int) -> pd.DataFrame:
    """Compare heads by cosine similarity of their raw attention patterns."""
    if layer < 0 or layer >= len(attentions):
        raise ValueError("Layer is outside the available attention range.")
    values = attentions[layer]
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("Attention tensors must have shape (1, heads, query, key).")
    flattened = F.normalize(values[0].float().flatten(start_dim=1), dim=-1)
    similarity = flattened @ flattened.T
    return pd.DataFrame(
        similarity.numpy(),
        index=[f"Head {head}" for head in range(values.shape[1])],
        columns=[f"Head {head}" for head in range(values.shape[1])],
    )


def cosine_by_layer(a: AnalysisResult, b: AnalysisResult) -> pd.DataFrame:
    count = min(len(a.hidden_states), len(b.hidden_states)) - 1
    return pd.DataFrame(
        {
            "layer": list(range(count)),
            "cosine_similarity": [
                float(
                    F.cosine_similarity(
                        a.hidden_states[i + 1][0, -1],
                        b.hidden_states[i + 1][0, -1],
                        dim=0,
                    )
                )
                for i in range(count)
            ],
        }
    )


@dataclass
class BatchResult:
    input_ids: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    logits: torch.Tensor


def analyse_batch(
    runtime: Runtime,
    prompts: list[str],
    max_length: int = 128,
) -> BatchResult:
    """Tokenize a list of prompts and run a single batched forward pass.

    Returns hidden states and logits for the whole batch without the
    single-prompt ``labels`` field. Use this for probe training data where
    you need hidden states for many prompts at once.
    """
    if not prompts:
        raise ValueError("At least one prompt is required.")
    batch = runtime.tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=True,
    ).to(runtime.device)
    with torch.inference_mode():
        outputs = runtime.model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = tuple(t.detach().float().cpu() for t in outputs.hidden_states)
    return BatchResult(
        input_ids=batch["input_ids"].detach().cpu(),
        hidden_states=hidden,
        logits=outputs.logits.detach().float().cpu(),
    )


def reduce_activations(
    result: AnalysisResult,
    layer: int,
    method: str = "PCA",
) -> pd.DataFrame:
    matrix = result.hidden_states[layer + 1][0].numpy()
    if len(matrix) < 2:
        raise ValueError("At least two tokens are required for projection.")
    if method == "UMAP":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("Install the 'umap' extra to use UMAP.") from exc
        n_neighbors = max(2, min(15, len(matrix) - 1))
        coords = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=42,
        ).fit_transform(matrix)
    else:
        coords = PCA(n_components=2).fit_transform(matrix)
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "token": result.labels,
            "position": np.arange(len(result.labels)),
        }
    )

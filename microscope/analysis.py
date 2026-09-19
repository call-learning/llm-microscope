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


def layer_norms(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": list(range(len(result.hidden_states) - 1)),
            "norm": [float(h[0, -1].norm()) for h in result.hidden_states[1:]],
        }
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


from __future__ import annotations

import pandas as pd
import torch

from .analysis import top_tokens
from .runtime import Runtime, tokenize


def _replace_final_token(output, replacement: torch.Tensor):
    if isinstance(output, tuple):
        hidden = output[0].clone()
        hidden[:, -1, :] = replacement.to(hidden.device, hidden.dtype)
        return (hidden, *output[1:])
    hidden = output.clone()
    hidden[:, -1, :] = replacement.to(hidden.device, hidden.dtype)
    return hidden


def patch_final_residual(
    runtime: Runtime,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    max_length: int = 256,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_batch = tokenize(runtime, source_prompt, max_length=max_length)
    target_batch = tokenize(runtime, target_prompt, max_length=max_length)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        captured["value"] = state[:, -1, :].detach().clone()

    handle = runtime.layers[layer].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            runtime.model(**source_batch, use_cache=False)
    finally:
        handle.remove()

    with torch.inference_mode():
        baseline = runtime.model(**target_batch, use_cache=False).logits[0, -1].float()

    def inject(_module, _inputs, output):
        return _replace_final_token(output, captured["value"])

    handle = runtime.layers[layer].register_forward_hook(inject)
    try:
        with torch.inference_mode():
            patched = runtime.model(**target_batch, use_cache=False).logits[0, -1].float()
    finally:
        handle.remove()

    return top_tokens(runtime, baseline.cpu(), k), top_tokens(runtime, patched.cpu(), k)


def gradient_x_input(
    runtime: Runtime,
    prompt: str,
    target_token_id: int | None = None,
    max_length: int = 256,
) -> tuple[pd.DataFrame, int]:
    batch = tokenize(runtime, prompt, max_length=max_length)
    embeddings = runtime.model.get_input_embeddings()(batch["input_ids"])
    embeddings = embeddings.detach().requires_grad_(True)
    runtime.model.zero_grad(set_to_none=True)
    outputs = runtime.model(
        inputs_embeds=embeddings,
        attention_mask=batch.get("attention_mask"),
        use_cache=False,
    )
    if target_token_id is None:
        target_token_id = int(outputs.logits[0, -1].argmax())
    score = outputs.logits[0, -1, target_token_id]
    score.backward()
    importance = (embeddings.grad * embeddings).sum(dim=-1).abs()[0]
    importance = importance / importance.max().clamp_min(1e-12)
    tokens = [
        runtime.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in batch["input_ids"][0]
    ]
    df = pd.DataFrame(
        {
            "position": range(len(tokens)),
            "token": [repr(token) for token in tokens],
            "importance": importance.detach().float().cpu().numpy(),
        }
    )
    return df, target_token_id


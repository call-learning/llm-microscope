from __future__ import annotations

from enum import Enum

import pandas as pd
import torch

from .analysis import top_tokens
from .runtime import Runtime, layer_modules, tokenize


class PatchTarget(Enum):
    """Which part of a decoder layer to patch or ablate."""

    RESIDUAL = "residual"
    ATTN_OUTPUT = "attn_output"
    MLP_OUTPUT = "mlp_output"


def _replace_final_token(output, replacement: torch.Tensor):
    if isinstance(output, tuple):
        hidden = output[0].clone()
        hidden[:, -1, :] = replacement.to(hidden.device, hidden.dtype)
        return (hidden, *output[1:])
    hidden = output.clone()
    hidden[:, -1, :] = replacement.to(hidden.device, hidden.dtype)
    return hidden


def _resolve_module(runtime: Runtime, layer: int, target: PatchTarget):
    """Return the module to hook for a given layer and target.

    For RESIDUAL, returns the whole decoder layer.
    For ATTN_OUTPUT / MLP_OUTPUT, returns the submodule; raises if the
    architecture does not follow the self_attn / mlp convention.
    """
    if target is PatchTarget.RESIDUAL:
        return runtime.layers[layer]
    attn, mlp = layer_modules(runtime.layers[layer])
    if target is PatchTarget.ATTN_OUTPUT:
        if attn is None:
            raise ValueError(
                f"Layer {layer} does not expose a self_attn/attention submodule. "
                "Module-level patching is unavailable for this architecture."
            )
        return attn
    if target is PatchTarget.MLP_OUTPUT:
        if mlp is None:
            raise ValueError(
                f"Layer {layer} does not expose an mlp submodule. "
                "Module-level patching is unavailable for this architecture."
            )
        return mlp
    raise ValueError(f"Unknown patch target: {target}")


def _capture_for_target(
    runtime: Runtime,
    source_prompt: str,
    target: PatchTarget,
    max_length: int = 128,
    head_idx: int = 0,
) -> dict[int, torch.Tensor]:
    """Run one source forward and capture the final-token state at every layer
    for the specified patch target.

    For RESIDUAL / MLP_OUTPUT: captures from hidden states (shape ``(hidden,)``).
    For ATTN_OUTPUT: captures the attention module's output via a forward hook
    (shape ``(hidden,)`` — the projected attention output after ``o_proj``).
    """
    batch = tokenize(runtime, source_prompt, max_length=max_length)

    if target is PatchTarget.ATTN_OUTPUT:
        # Capture via forward hooks on the attention module
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_idx in range(len(runtime.layers)):
            attn, _ = layer_modules(runtime.layers[layer_idx])
            if attn is None:
                raise ValueError(
                    f"Layer {layer_idx} does not expose a self_attn/attention submodule. "
                    "Attention-output patching is unavailable for this architecture."
                )

            def make_capture(li: int):
                def capture(_module, _inputs, output):
                    state = output[0] if isinstance(output, tuple) else output
                    captured[li] = state[0, -1, :].detach().clone()
                return capture

            handle = attn.register_forward_hook(make_capture(layer_idx))
            hooks.append(handle)

        try:
            with torch.inference_mode():
                runtime.model(**batch, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return captured
    else:
        # RESIDUAL or MLP_OUTPUT: use output_hidden_states
        with torch.inference_mode():
            outputs = runtime.model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        states: dict[int, torch.Tensor] = {}
        for layer_idx in range(len(runtime.layers)):
            hidden = outputs.hidden_states[layer_idx + 1]
            states[layer_idx] = hidden[0, -1, :].detach().clone()
        return states


def _patched_forwards(
    runtime: Runtime,
    target_batch: dict,
    source_states: dict[int, torch.Tensor],
    target: PatchTarget = PatchTarget.RESIDUAL,
    max_length: int = 128,
    head_idx: int = 0,
) -> list[torch.Tensor]:
    """Run one patched target forward per layer, returning the final-position logits.

    Registers an injection hook on the target module for each layer, runs one
    target forward, reads ``logits[0, -1]``, removes the hook, and returns the
    list of patched logits (one per layer).

    For ATTN_OUTPUT, hooks the ``o_proj`` submodule and replaces the
    final-token slice of its input (the un-projected attention output,
    shape ``(batch, n_heads, seq, head_dim)``) with the source value
    reshaped to match. This allows per-query-head patching.
    """
    n_layers = len(runtime.layers)
    patched_logits: list[torch.Tensor] = []

    for layer_idx in range(n_layers):
        replacement = source_states[layer_idx]

        if target is PatchTarget.ATTN_OUTPUT:
            attn, _ = layer_modules(runtime.layers[layer_idx])
            if attn is None:
                raise ValueError(
                    f"Layer {layer_idx} does not expose a self_attn/attention submodule."
                )

            def make_attn_inject(rep: torch.Tensor):
                def inject_attn(_module, _inputs, output):
                    # output[0] is the projected attention output:
                    # shape (batch, seq, hidden)
                    return _replace_final_token(output, rep)
                return inject_attn

            handle = attn.register_forward_hook(
                make_attn_inject(replacement)
            )
        else:
            module = _resolve_module(runtime, layer_idx, target)

            def make_inject(rep: torch.Tensor):
                def inject(_module, _inputs, output):
                    return _replace_final_token(output, rep)
                return inject

            handle = module.register_forward_hook(make_inject(replacement))

        try:
            with torch.inference_mode():
                out = runtime.model(**target_batch, use_cache=False)
            patched_logits.append(out.logits[0, -1].float().clone())
        finally:
            handle.remove()

    return patched_logits


def patching_score_curve(
    runtime: Runtime,
    source_prompt: str,
    target_prompt: str,
    target: PatchTarget = PatchTarget.RESIDUAL,
    max_length: int = 128,
    k: int = 10,
    head_idx: int = 0,
) -> pd.DataFrame:
    """Compute the patching score curve for a source→target prompt pair.

    Runs 1 source forward (capturing all layer hidden states), 1 baseline
    target forward, and N patched target forwards (one per layer), then
    assembles a DataFrame with per-layer top-token detail and delta columns.

    Args:
        head_idx: For ATTN_OUTPUT target, which query head to patch (0-indexed).
                  Ignored for RESIDUAL / MLP_OUTPUT targets.

    Returns a DataFrame with columns:
        layer, baseline_top1_token, patched_top1_token,
        baseline_top1_prob, patched_top1_prob, delta_logit, delta_prob
    """
    target_batch = tokenize(runtime, target_prompt, max_length=max_length)

    # 1. Source capture (1 forward)
    source_states = _capture_for_target(
        runtime, source_prompt, target, max_length, head_idx
    )

    # 2. Baseline target (1 forward)
    with torch.inference_mode():
        baseline_logits = runtime.model(
            **target_batch, use_cache=False
        ).logits[0, -1].float()

    baseline_probs = baseline_logits.softmax(dim=-1)
    baseline_top1_id = int(baseline_logits.argmax())
    baseline_top1_prob = float(baseline_probs[baseline_top1_id])
    baseline_top1_token = runtime.tokenizer.decode(
        [baseline_top1_id], clean_up_tokenization_spaces=False
    )

    # 3. Patched forwards (N forwards)
    patched_logits_list = _patched_forwards(
        runtime, target_batch, source_states, target, max_length, head_idx
    )

    rows = []
    for layer_idx in range(len(runtime.layers)):
        patched = patched_logits_list[layer_idx]
        patched_probs = patched.softmax(dim=-1)
        patched_top1_id = int(patched.argmax())
        patched_top1_prob = float(patched_probs[patched_top1_id])
        patched_top1_token = runtime.tokenizer.decode(
            [patched_top1_id], clean_up_tokenization_spaces=False
        )

        # Delta on the *baseline's* top-1 token (stable reference)
        delta_logit = float(patched[baseline_top1_id] - baseline_logits[baseline_top1_id])
        delta_prob = float(patched_probs[baseline_top1_id] - baseline_probs[baseline_top1_id])

        rows.append(
            {
                "layer": layer_idx,
                "baseline_top1_token": baseline_top1_token,
                "patched_top1_token": patched_top1_token,
                "baseline_top1_prob": round(baseline_top1_prob, 6),
                "patched_top1_prob": round(patched_top1_prob, 6),
                "delta_logit": round(delta_logit, 6),
                "delta_prob": round(delta_prob, 6),
            }
        )

    return pd.DataFrame(rows)


def patching_effect_matrix(
    runs: list[dict[str, object]],
    metric: str,
) -> pd.DataFrame:
    """Assemble compatible patch-curve runs into a layer-by-target matrix."""
    if not runs:
        raise ValueError("At least one patching run is required.")
    rows = []
    for run in runs:
        curve = run["curve"]
        if not isinstance(curve, pd.DataFrame) or metric not in curve:
            raise ValueError(f"Patching run does not contain metric {metric!r}.")
        values = curve.set_index("layer")[metric]
        rows.append(
            {
                "patch target": run["patch_target_display"],
                **{int(layer): value for layer, value in values.items()},
            }
        )
    return pd.DataFrame(rows).set_index("patch target")


def ablation_effect_curve(
    runtime: Runtime,
    prompt: str,
    target: PatchTarget = PatchTarget.RESIDUAL,
    max_length: int = 128,
    target_token_id: int | None = None,
    head_idx: int = 0,
) -> pd.DataFrame:
    """Measure the selected-token effect of zero ablation at every layer.

    The unmodified prompt is evaluated once. Each subsequent forward removes
    only the final-token output of one selected component, so results are
    comparable across layers and targets.
    """
    batch = tokenize(runtime, prompt, max_length=max_length)
    with torch.inference_mode():
        baseline = runtime.model(**batch, use_cache=False).logits[0, -1].float()
    if target_token_id is None:
        target_token_id = int(baseline.argmax())
    if target_token_id < 0 or target_token_id >= baseline.numel():
        raise ValueError("Target token ID is outside the vocabulary.")

    rows = []
    baseline_value = float(baseline[target_token_id])
    baseline_probability = float(baseline.softmax(dim=-1)[target_token_id])
    for layer_idx in range(len(runtime.layers)):
        if target is PatchTarget.ATTN_OUTPUT:
            module, _ = layer_modules(runtime.layers[layer_idx])
        elif target is PatchTarget.MLP_OUTPUT:
            _, module = layer_modules(runtime.layers[layer_idx])
        else:
            module = runtime.layers[layer_idx]
        if module is None:
            raise ValueError(f"Layer {layer_idx} does not expose the selected component.")

        if target is PatchTarget.ATTN_OUTPUT and hasattr(module, "o_proj"):
            # The input to o_proj is the per-head attention result in common
            # decoder implementations. Zero only one head before projection.
            projection = module.o_proj

            def ablate_head(_module, inputs):
                if not inputs:
                    raise ValueError("Attention projection did not expose head inputs.")
                value = inputs[0].clone()
                if value.ndim == 4:
                    if head_idx < 0 or head_idx >= value.shape[2]:
                        raise ValueError(f"Attention head {head_idx} is outside the available range.")
                    value[:, -1, head_idx, :] = 0
                elif value.ndim == 3:
                    n_heads = getattr(runtime.model.config, "num_attention_heads", 0)
                    if not n_heads or value.shape[-1] % n_heads:
                        raise ValueError("Attention projection input cannot be split into heads.")
                    head_dim = value.shape[-1] // n_heads
                    if head_idx < 0 or head_idx >= n_heads:
                        raise ValueError(f"Attention head {head_idx} is outside the available range.")
                    value[:, -1, head_idx * head_dim:(head_idx + 1) * head_dim] = 0
                else:
                    raise ValueError("Attention projection input has an unsupported shape.")
                return (value, *inputs[1:])

            handle = projection.register_forward_pre_hook(ablate_head)
        else:
            def ablate(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                replacement = torch.zeros_like(hidden[:, -1:, :])
                return _replace_final_token(output, replacement)

            handle = module.register_forward_hook(ablate)
        try:
            with torch.inference_mode():
                ablated = runtime.model(**batch, use_cache=False).logits[0, -1].float()
        finally:
            handle.remove()
        probability = float(ablated.softmax(dim=-1)[target_token_id])
        rows.append({
            "layer": layer_idx,
            "target": target.value,
            "head_idx": head_idx if target is PatchTarget.ATTN_OUTPUT else None,
            "target_token_id": target_token_id,
            "baseline_logit": baseline_value,
            "ablated_logit": float(ablated[target_token_id]),
            "delta_logit": float(ablated[target_token_id] - baseline_value),
            "baseline_probability": baseline_probability,
            "ablated_probability": probability,
            "delta_probability": probability - baseline_probability,
        })
    return pd.DataFrame(rows)


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


def mean_ablation(
    runtime: Runtime,
    source_prompt: str,
    target_prompt: str,
    layer: int,
    target: PatchTarget = PatchTarget.RESIDUAL,
    mode: str = "zero",
    max_length: int = 128,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ablate a selected target at a layer by zeroing or mean-replacing its value.

    Args:
        mode: "zero" sets the final-token value to 0.
              "mean" sets it to the mean of the source-run value.
              For single-prompt runs, mean ≈ patch (same vector).

    Returns:
        (baseline_df, ablated_df) — top-token DataFrames for the un-ablated
        and ablated target runs.
    """
    if mode not in ("zero", "mean"):
        raise ValueError(f"Unknown ablation mode: {mode!r}. Use 'zero' or 'mean'.")

    target_batch = tokenize(runtime, target_prompt, max_length=max_length)

    # Capture source value (for mean mode)
    if mode == "mean":
        source_states = _capture_for_target(runtime, source_prompt, target, max_length)
        source_value = source_states[layer]
        mean_value = source_value.mean()  # scalar mean of the source vector
    else:
        mean_value = None

    # Baseline (1 forward)
    with torch.inference_mode():
        baseline_logits = runtime.model(
            **target_batch, use_cache=False
        ).logits[0, -1].float()

    # Ablated forward (1 forward)
    if target is PatchTarget.ATTN_OUTPUT:
        attn, _ = layer_modules(runtime.layers[layer])
        if attn is None:
            raise ValueError(
                f"Layer {layer} does not expose a self_attn/attention submodule."
            )

        def inject_attn_ablate(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            if mode == "zero":
                hidden[0, -1, :] = 0
            else:
                hidden[0, -1, :] = float(mean_value)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = attn.register_forward_hook(inject_attn_ablate)
    else:
        module = _resolve_module(runtime, layer, target)

        def inject_ablate(_module, _inputs, output):
            if mode == "zero":
                replacement = torch.zeros(
                    1, 1, output[0].shape[-1] if isinstance(output, tuple) else output.shape[-1],
                    device=output[0].device if isinstance(output, tuple) else output.device,
                )
            else:
                replacement = torch.full(
                    (1, 1, output[0].shape[-1] if isinstance(output, tuple) else output.shape[-1]),
                    float(mean_value),
                    device=output[0].device if isinstance(output, tuple) else output.device,
                )
            return _replace_final_token(output, replacement)

        handle = module.register_forward_hook(inject_ablate)

    try:
        with torch.inference_mode():
            ablated_logits = runtime.model(
                **target_batch, use_cache=False
            ).logits[0, -1].float()
    finally:
        handle.remove()

    return (
        top_tokens(runtime, baseline_logits.cpu(), k),
        top_tokens(runtime, ablated_logits.cpu(), k),
    )


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
    importance = (embeddings.grad * embeddings).sum(dim=-1)[0]
    importance = importance / importance.abs().max().clamp_min(1e-12)
    tokens = [
        runtime.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in batch["input_ids"][0]
    ]
    df = pd.DataFrame(
        {
            "position": range(len(tokens)),
            "token": [repr(token) for token in tokens],
            "importance": importance.detach().float().cpu().numpy(),
            "magnitude": importance.detach().float().cpu().abs().numpy(),
            "method": "Gradient x Input",
        }
    )
    return df, target_token_id


def integrated_gradients(
    runtime: Runtime,
    prompt: str,
    target_token_id: int | None = None,
    max_length: int = 256,
    n_steps: int = 16,
) -> tuple[pd.DataFrame, int]:
    """Compute optional Captum Integrated Gradients for one output token."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError(
            "Integrated Gradients requires the optional 'captum' extra."
        ) from exc
    if n_steps < 2:
        raise ValueError("Integrated Gradients requires at least 2 steps.")

    batch = tokenize(runtime, prompt, max_length=max_length)
    embeddings = runtime.model.get_input_embeddings()(batch["input_ids"]).detach()
    with torch.inference_mode():
        baseline = torch.zeros_like(embeddings)
        initial_outputs = runtime.model(
            inputs_embeds=embeddings,
            attention_mask=batch.get("attention_mask"),
            use_cache=False,
        )
    if target_token_id is None:
        target_token_id = int(initial_outputs.logits[0, -1].argmax())

    def forward(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        outputs = runtime.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits[:, -1, target_token_id]

    runtime.model.zero_grad(set_to_none=True)
    attributions = IntegratedGradients(forward).attribute(
        embeddings,
        baselines=baseline,
        additional_forward_args=(batch.get("attention_mask"),),
        n_steps=n_steps,
    )
    signed = attributions.sum(dim=-1)[0]
    signed = signed / signed.abs().max().clamp_min(1e-12)
    tokens = [
        runtime.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in batch["input_ids"][0]
    ]
    df = pd.DataFrame(
        {
            "position": range(len(tokens)),
            "token": [repr(token) for token in tokens],
            "importance": signed.detach().float().cpu().numpy(),
            "magnitude": signed.detach().float().cpu().abs().numpy(),
            "method": "Integrated Gradients",
        }
    )
    return df, target_token_id

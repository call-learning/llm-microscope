## Context

The current patching implementation (`microscope/interventions.py::patch_final_residual`) captures the final-token residual state from a *single* layer of a source forward, then runs one un-patched target forward and one patched target forward. It registers a `forward_hook` on `runtime.layers[layer]` to capture and another to inject. The hook targets the whole-layer output (the residual stream after that layer).

The `Runtime` dataclass (`microscope/runtime.py`) already holds `layers` (the decoder layer list), `final_norm`, `lm_head`, and `device`. It does **not** currently hold references to per-layer submodules (attention / MLP).

Qwen3-1.7B (the default model) has 28 layers, `hidden_size=2048`, 16 query heads, 8 KV heads (GQA), and `tie_word_embeddings=true`. The Llama/Qwen/Mistral layer convention is `layer.self_attn` (attention block) and `layer.mlp` (MLP block), each returning a tensor that is added to the residual stream.

## Goals / Non-Goals

**Goals:**
- Compute a per-layer patching score curve: how the target's next-token prediction changes when the final-token state is patched at each layer, for a given source→target pair.
- Extend patching from the residual stream to attention-output and MLP-output targets.
- Add mean ablation (zero / mean-replace) on the same targets.
- Expose attention-output patching at the query-head level (16 for Qwen3-1.7B).
- Run all of this with a deduplicated forward loop: 1 source forward + 1 baseline target forward + N patched forwards.
- Preserve the existing single-layer residual patching behaviour as the default / residual target.

**Non-Goals:**
- Path patching (patching the attention *and* MLP outputs simultaneously, or patching across positions).
- Activation patching at non-final-token positions.
- A steering / activation-addition UI (that is the follow-on change to `add-linear-probe`).
- Making the submodule introspection generic across all Hugging Face architectures. The scope limit is explicit: module-level targets work on the `self_attn`/`mlp` convention; residual works everywhere.

## Decisions

### 1. How to capture hidden states at all layers from a single forward

**Decision:** Use `output_hidden_states=True` on the source forward, not per-layer hooks. The existing `analyse()` already does this — `outputs.hidden_states` is a tuple of `(batch, seq, hidden)` tensors, one per layer boundary (entry 0 = embedding output, entry L+1 = after layer L). We reuse that pattern for the source capture: one forward, grab `hidden_states[L+1][:, -1, :]` for every L.

**Rationale:** Hooks are stateful and awkward to register/remove N times in a loop; `output_hidden_states` is a single model kwarg that returns everything at once. It is already the pattern used by `analyse()`, so it is consistent with the codebase.

**Alternative considered:** Registering a capture hook on each layer in a loop. Rejected: more code, more hook management, and the model still runs once either way — no forward-pass savings.

### 2. How to run the patched forwards

**Decision:** For each layer L, register an injection hook on the target module for that layer, run one target forward, read `logits[0, -1]`, remove the hook. The injection hook replaces the module's output (or the final-token slice of it) with the captured value.

**Rationale:** This is the minimal change to the existing `patch_final_residual` injection mechanism. Each patched forward is independent; no state needs to persist between layers.

**Alternative considered:** A single forward with a "patch at every layer simultaneously" approach. Rejected: that would patch all layers at once, which is not what a score curve needs — we need *one layer at a time* to isolate each layer's causal contribution.

### 3. How to select the injection target (residual vs attention-output vs MLP-output)

**Decision:** Introduce a target enum: `RESIDUAL`, `ATTN_OUTPUT`, `MLP_OUTPUT`. For `RESIDUAL`, hook `runtime.layers[L]` (whole layer) as today. For `ATTN_OUTPUT`, hook the `layer.self_attn` submodule. For `MLP_OUTPUT`, hook `layer.mlp`. The injection logic replaces the module's output (or the final-token slice) with the captured value.

**Rationale:** `forward_hook` on a submodule intercepts that submodule's output, which is exactly the attention or MLP contribution to the residual stream at that layer. This is the standard "activation patching" target in the mechanistic-interpretability literature.

**GQA detail for ATTN_OUTPUT:** The attention module's output is a tensor of shape `(batch, seq, hidden)` — it is the *summed* output of all query heads, already projected back to `hidden_size`. There is no per-head dimension in the output tensor. To expose per-query-head patching, we must hook the attention output **before** the `o_proj` (output projection) that collapses the heads. This requires hooking the `attn_output` tensor inside the `self_attn` module, not the module's final output. The exact hook point depends on the attention implementation (e.g. `Qwen3Attention` returns the projected output; the un-projected `(batch, n_heads, seq, head_dim)` tensor is an intermediate). This is the one place the implementation needs to be architecture-aware.

**Alternative considered:** Patching at the KV-head level (8 heads). Rejected: the KV heads are shared across query heads in GQA; patching a KV head affects multiple query heads and is not independently controllable at the 16-head granularity.

### 4. Submodule introspection with graceful fallback

**Decision:** Add a helper in `runtime.py` (or a new `microscope/introspection.py`) that, given a layer module, returns `(attn_module, mlp_module)` if they follow the `self_attn` / `mlp` convention, or `None` if they don't. The UI checks this before enabling the module-target selector; if `None`, only `RESIDUAL` is selectable and a caption explains why.

**Rationale:** The architecture-sniffing problem is already present in `runtime.py::decoder_layers` and `final_norm` (hardcoded candidate paths). Extending it to per-layer submodules follows the same pattern. The scope limit is explicit: we do not attempt to make this generic across all HF architectures; we support the common convention and degrade gracefully.

**Candidates checked (in order):**
1. `layer.self_attn` / `layer.mlp` — Llama, Qwen, Mistral
2. `layer.h` — GPT-2 (attention + MLP are inside, not separate submodules) → module targets unavailable
3. `layer.attention` / `layer.mlp` — some GPT-NeoX variants

### 5. Mean ablation: zero vs mean

**Decision:** Two sub-modes: `zero` (set the target's final-token value to 0) and `mean` (set it to the mean of the source-run value across the batch dimension — for a single-prompt run this is just the source value, so mean and the captured value are the same; the distinction matters when the source run is a batch). For a single-prompt run, "mean ablation" is effectively "replace with the source value," which is the same as patching. The zero ablation is the more informative mode. We implement both but document that for single-prompt runs, mean ≈ patch.

**Rationale:** Mean ablation is the standard ablation in the literature when you don't have a clean source. For a single prompt, zero ablation is the cleaner test ("does this component do anything?").

### 6. Where the curve lives in the data flow

**Decision:** The score curve is a **separate, on-demand computation** — it does not modify `AnalysisResult` or the existing `analyse()` path. It is a new function in `interventions.py` (or a new `microscope/patching.py`) that takes `(runtime, source_prompt, target_prompt, target, max_length)` and returns a `pd.DataFrame` with one row per layer. The app calls it behind a spinner, like the existing patching button.

**Rationale:** Keeps the existing `AnalysisResult` data flow untouched. The curve is a multi-forward experiment, not a single-forward analysis; it belongs with the interventions, not the analysis.

## Risks / Trade-offs

- **[Per-layer attention-output hook point is architecture-specific]** → The exact intermediate tensor to hook (before `o_proj`) differs between Qwen3, Llama, and Mistral attention implementations. Mitigation: start with Qwen3 (the default model), test on Llama, and add a clear "not supported" message for others. Do not attempt to support all attention variants in the first pass.
- **[28 forwards is ~30–60s on a 3090]** → This is acceptable for an on-demand experiment (the existing patching button already runs multi-forward experiments). The spinner communicates the wait. No background-job machinery needed.
- **[Memory: 28 hidden-state tensors held on CPU]** → Each is `(1, seq, 2048)` float32 ≈ 1 MB for seq=128. 28 of them ≈ 28 MB. Negligible. The model weights are the dominant memory consumer and are unchanged.
- **[Mean ablation on single-prompt runs is equivalent to patching]** → Document this explicitly in the UI so the user doesn't expect a different result. The zero ablation is the distinct, useful mode.

## Migration Plan

None — this is a new capability, not a modification to existing behaviour. The existing `patch_final_residual` function is preserved (it becomes the `RESIDUAL` target). No data migration, no API break.

## Resolved Decisions

- **Score curve reports both logit delta and probability delta.** The per-layer table shows both columns (`baseline_logit`, `patched_logit`, `delta_logit`, `baseline_prob`, `patched_prob`, `delta_prob`). The curve plot shows both with a toggle (default: logit delta, which is more sensitive; probability delta as the alternative view).
- **Per-layer top-token table is included** alongside the curve. Columns: `layer`, `baseline_top1_token`, `patched_top1_token`, `baseline_top1_prob`, `patched_top1_prob`, `delta_logit`, `delta_prob`.

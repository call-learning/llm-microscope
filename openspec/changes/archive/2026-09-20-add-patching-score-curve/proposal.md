## Why

The current patching tab can only copy the final-token residual state at a *single* layer from one prompt to another. The interesting interpretability question it gestures at — "at which layer does the model's prediction for the target prompt change?" — is not answerable without sweeping every layer. This change turns a single-layer poke into a causal curve, and extends patching from the residual stream to the module-level targets (attention output, MLP output) that carry the signal.

## What Changes

- Add a **patching score curve** view: for a source→target prompt pair, report how the target's next-token prediction changes when the final-token state is patched at *every* layer, plotted as a curve over layers with per-layer top-token detail.
- Run the sweep with a **deduplicated loop**: one source forward (capturing hidden states at all layers), one baseline target forward, then one patched forward per layer — ~30 forward passes total for a 28-layer model, instead of the ~3× cost of re-capturing and re-baselining inside the loop.
- Add a **patch target selector**: patch the residual stream (current behaviour, architecture-agnostic) *or* the attention-block output *or* the MLP output. Module-level targets are supported for the common `self_attn`/`mlp` layer convention (Llama/Qwen/Mistral family) and degrade gracefully with a clear message on other architectures.
- Add a **mean ablation** mode: zero or mean-replace the selected target (no clean source prompt required), to answer "does this component do anything at all?"
- GQA-aware: attention-output patching is exposed at the query-head level (e.g. 16 query heads for Qwen3-1.7B) rather than the KV-head level (8), matching what is independently patchable on the output side.
- Present results synchronously behind a spinner, matching the existing on-demand experiment pattern.

## Capabilities

### New Capabilities
- `activation-patching`: causal intervention views over the decoder — residual-stream patching, module-level (attention-output / MLP-output) patching, and mean ablation — including the per-layer patching score curve and GQA-aware target selection.

### Modified Capabilities
<!-- None. No existing capability is being modified; the project has no baseline specs yet. -->

## Impact

- `microscope/interventions.py` — new/extended patching and ablation functions; existing `patch_final_residual` behaviour preserved as the residual target.
- `microscope/runtime.py` — per-layer submodule introspection (attention / MLP module) with graceful fallback when the convention is not found.
- `app.py` — Patching tab extended with target selector, mean-ablation toggle, and score-curve plot.
- No new runtime dependencies (PyTorch, pandas, and plotly are already present).
- No change to the fat-forward-pass `AnalysisResult` data flow; the curve is a separate on-demand computation.
- Memory: stays within the documented 12 GB budget — the sweep reuses captured CPU tensors and runs one patched forward at a time.

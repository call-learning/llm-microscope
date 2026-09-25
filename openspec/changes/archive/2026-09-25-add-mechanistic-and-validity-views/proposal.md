# Proposal

## Why

The workbench already exposes layer outputs, attention patterns, attribution, probes, and activation patching, but users still lack a coherent way to distinguish descriptive patterns from causal mechanisms and from reliable model confidence. This change adds deeper layer interpretation and empirical validity analysis while preserving the existing Hugging Face/Plotly path and its architecture fallbacks.

## What Changes

- Add tuned-lens analysis alongside the existing raw logit lens.
- Add residual-stream and component contribution views for embeddings, attention, and MLP outputs.
- Add stronger causal validation views, including component/path comparisons and probe-versus-intervention comparisons where supported.
- Add a validity and confidence workspace covering calibration, uncertainty, perturbation sensitivity, and answer consistency.
- Add optional integration boundaries for TransformerLens and SAELens-backed mechanistic or sparse-feature views without making them core dependencies.
- Document that confidence, attention, attribution, probes, and lens outputs are evidence with different limitations, not proof of factual validity or a literal reasoning trace.

## Capabilities

### New Capabilities

- `validity-analysis`: Calibration, uncertainty, consistency, and controlled perturbation views for model outputs.
- `mechanistic-analysis`: Residual/component decomposition and causal validation views over layers, heads, and MLPs.

### Modified Capabilities

- `analysis-visualizations`: Add tuned-lens and component-aware layer visualisations with explicit interpretation limits.
- `advanced-tool-integrations`: Define safe optional adapters for mechanistic tracing and sparse autoencoder features.

## Impact

- Affected application pages and analysis helpers in `app.py` and `microscope/analysis.py`.
- New inference and experiment orchestration for calibration, perturbation, decomposition, and optional tuned-lens/SAE providers.
- Additional optional dependencies may be introduced only with isolated extras and compatibility checks.
- New focused unit tests and, where practical, small-model integration tests; no change to the existing base model-loading contract.

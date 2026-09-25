# Proposal

## Why

The project now installs Tuned Lens, BertViz, and CircuitsVis directly, but the Toolbox still treats them as unintegrated candidates, while SAELens and Pyvene remain undiscoverable without manual installation. Adapters should turn installed packages into explicit, isolated capabilities without making third-party wrappers or browser renderers a requirement for native analysis.

## What Changes

- Add adapter contracts and compatibility probes for Tuned Lens, BertViz, CircuitsVis, TransformerLens, NNsight, SAELens, and Pyvene.
- Render BertViz and CircuitsVis outputs through optional Streamlit-compatible views with native Plotly fallbacks.
- Use TransformerLens and NNsight for explicit tracing/intervention experiments without replacing the core Hugging Face runtime.
- Add optional SAELens and Pyvene dependency extras and report missing-package/install instructions.
- Separate installed, compatible, adapter-ready, artefact-ready, and rendered states in the Toolbox.

## Capabilities

### New Capabilities

<!-- None; this extends the existing optional integration and visualisation contracts. -->

### Modified Capabilities

- `advanced-tool-integrations`: define adapter readiness, provider probes, and failure isolation for all optional tools.
- `analysis-visualizations`: define optional provider rendering and native fallback behaviour.

## Impact

- New lazy adapters under `microscope/` and Toolbox compatibility logic.
- Optional dependency metadata and lockfile updates for SAELens and Pyvene only.
- Streamlit rendering paths must tolerate missing frontend assets and provider API changes.
- Native Hugging Face views remain the baseline and must continue working when any adapter fails.

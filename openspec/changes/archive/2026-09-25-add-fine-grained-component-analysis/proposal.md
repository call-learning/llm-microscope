# Proposal

## Why

The current Mechanistic view identifies influential layers, module outputs, and attention heads, but it does not show the finer-grained units that compose those outputs. Users need a safe way to inspect MLP neurons, attention Q/K/V signals, causal paths, and optional sparse features without confusing visualization with a complete explanation.

## What Changes

- Add neuron-level MLP activation and ablation views.
- Add attention Q/K/V and per-head signal summaries where the architecture exposes them.
- Add path-level intervention views for selected component-to-component routes.
- Add an optional SAE feature explorer with explicit model, layer, and artefact compatibility.
- Preserve native fallbacks when architecture details or optional sparse-feature artefacts are unavailable.

## Capabilities

### New Capabilities

- `fine-grained-components`: Neuron, Q/K/V, path, and sparse-feature inspection.

### Modified Capabilities

- `mechanistic-analysis`: Extend causal component analysis to neuron and path targets.
- `advanced-tool-integrations`: Add safe SAE provider and artefact status reporting.

## Impact

- New hook and intervention helpers in `microscope/interventions.py` and analysis helpers for component tensors.
- New Mechanistic UI controls and charts in `app.py`.
- Optional SAELens integration remains isolated from base dependencies.
- Additional tests are required for hook cleanup, architecture fallbacks, and artefact mismatches.

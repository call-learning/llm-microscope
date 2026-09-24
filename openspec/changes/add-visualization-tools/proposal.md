# Proposal

## Why

The workbench already provides native Plotly visualisations and reports several optional research integrations, but four visualisation-oriented tools are only listed as unintegrated candidates. A clear specification is needed before deciding which integrations can safely add value without making the core application dependent on them.

## What Changes

- Define the supported scope for evaluating SAELens, Pyvene, BertViz, and CircuitsVis as optional visualisation integrations.
- Require the Toolbox to distinguish installed, compatible, and integrated states for each candidate tool.
- Define safe behaviour when an optional package is unavailable, incompatible, or has no application adapter.
- Define visualisation requirements for token, attention, circuit, activation, or sparse-feature output where a tool is adopted.
- Keep the existing Hugging Face and Plotly visualisations usable without any candidate integration.

## Capabilities

### New Capabilities

<!-- None. The change extends existing optional-tool and visualisation contracts. -->

### Modified Capabilities

- `advanced-tool-integrations`: clarify lifecycle and status reporting for optional visualisation tools and their adapters.
- `analysis-visualizations`: define requirements for safely exposing additional interactive visualisation providers.

## Impact

- OpenSpec requirements for optional visualisation integrations and their user-facing status reporting.
- Potential future changes to `microscope/toolbox.py`, `app.py`, and optional dependency metadata; this proposal does not itself add dependencies or adapters.
- No change to the core analysis path or existing visualisation behaviour.

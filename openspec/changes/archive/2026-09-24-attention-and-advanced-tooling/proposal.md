## Why

The current Attention tab shows one raw heatmap for one layer and head, while the Toolbox only reports whether a few optional libraries are installed. Users cannot easily compare heads, summarise attention behavior, inspect rollout-style aggregates, or understand what additional research integrations can and cannot provide.

## What Changes

- Improve attention heatmaps with readable token-position labeling, hover details, and head-level summary metrics.
- Add attention entropy, maximum-attended-position, head comparison, and carefully labeled rollout-style views.
- Add optional head similarity or clustering views based on an explicitly documented representation of head behavior.
- Expand the Toolbox to report compatibility and capabilities for TransformerLens, NNsight, Captum, and any newly supported optional integrations.
- Evaluate optional integrations such as CircuitsVis/BertViz, SAELens, and Pyvene without making them mandatory for the core application.
- Add interpretation guidance that distinguishes attention correlation from causal influence and marks compatibility limitations.

## Capabilities

### New Capabilities

- `attention-analysis`: Interactive attention summaries, comparisons, and aggregate views across layers and heads.
- `advanced-tool-integrations`: Capability detection, compatibility reporting, and optional integration boundaries for external interpretability tools.

### Modified Capabilities

- None.

## Impact

- Affected the Attention and Toolbox tabs, `microscope/toolbox.py`, analysis helpers, and centralized documentation.
- Existing eager attention capture remains required for raw attention matrices; optimized kernels must continue to show a clear unavailable state.
- New external libraries must remain optional and isolated so they cannot block core Hugging Face-based analysis.
- Attention rollout and clustering require documented methodological choices and should not be presented as definitive explanations.

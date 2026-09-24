## 1. Attention Presentation

- [x] 1.1 Add readable position-based axes, token hover data, and query/key labels to the existing attention heatmap; verify short and long prompts remain inspectable.
- [x] 1.2 Add per-head entropy and maximum-attended-position summaries with layer/head identity; verify summary values against manually computed attention rows.
- [x] 1.3 Add head comparison controls and a compact comparison chart; verify selected heads remain distinguishable across layers.

## 2. Aggregate Attention Views

- [x] 2.1 Define and implement the documented rollout aggregation and normalization behavior; verify the aggregate has expected dimensions and remains separate from raw attention.
- [x] 2.2 Add rollout controls, assumptions, and descriptive-not-causal guidance; verify unsupported attention shapes show an explanation instead of failing.
- [x] 2.3 Add optional raw-pattern similarity support for head comparison and defer clustering output until its feature representation and distance metric are documented; verify no unsupported cluster interpretation is displayed.
- [x] 2.4 Preserve optimized-kernel handling and non-attention tab availability when attention matrices are absent; verify SDPA/FlashAttention produce a clear Attention warning.

## 3. Optional Tool Registry

- [x] 3.1 Extend the Toolbox registry with package status, capability description, and optional compatibility probe metadata; verify existing TransformerLens, NNsight, Captum, and UMAP statuses remain accurate.
- [x] 3.2 Add explicit user-triggered compatibility probes for supported integrations with isolated imports and actionable failures; verify a failed probe does not crash the app or block core analysis.
- [x] 3.3 Document SAELens, Pyvene, BertViz, and CircuitsVis as evaluation candidates without installing or executing them by default; verify their absence is reported as optional rather than an error.
- [x] 3.4 Add model, device, and memory limitations to Toolbox documentation; verify capability descriptions distinguish tracing, intervention, attribution, sparse features, and visualization.

## 4. Validation

- [x] 4.1 Add focused tests for attention summaries, rollout dimensions/normalization, unavailable attention behavior, and optional-tool status handling; verify the project test command passes.
- [x] 4.2 Run Python compilation, documentation/import checks, and strict OpenSpec validation for `attention-and-advanced-tooling`; verify no syntax or specification errors remain.
- [x] 4.3 Run a Streamlit smoke test with eager and optimized attention modes and with optional integrations absent; verify all core tabs remain usable.

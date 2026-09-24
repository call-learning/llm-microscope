## Context

See `proposal.md` and the capability specs for the user-facing behavior. The current Attention tab receives raw per-layer attention matrices only when the eager kernel is selected and renders one heatmap for one head. The Toolbox detects Captum, UMAP, NNsight, and TransformerLens, but only TransformerLens has a compatibility probe. Optional packages must not affect the core Hugging Face analysis path.

## Goals / Non-Goals

**Goals:**

- Make the existing raw attention heatmap readable and comparable without changing the model forward contract.
- Derive head summaries from captured attention matrices and label all aggregation assumptions.
- Provide an explicitly descriptive rollout view rather than presenting attention as causal evidence.
- Separate optional-tool installation, compatibility, and capability reporting.
- Keep new integrations isolated behind optional imports and explicit user actions.

**Non-Goals:**

- No claim that attention rollout identifies a circuit or proves token-level causation.
- No mandatory BertViz, CircuitsVis, SAELens, Pyvene, TransformerLens, or NNsight dependency.
- No automatic download or execution of third-party tools on application startup.
- No head clustering method without a stable, documented feature representation.

## Decisions

### 1. Keep Plotly as the core attention renderer

Improve the existing Plotly heatmap with position-aware axes, token hover data, and summary charts. This avoids introducing a second rendering model and keeps the core view available in the current Streamlit environment.

Alternative considered: embed BertViz or CircuitsVis. Those may offer richer attention interactions, but they add packaging and compatibility complexity and should be evaluated as optional adapters rather than become the core path.

### 2. Compute head summaries from raw attention tensors

For each selected layer and head, calculate entropy over attended key positions and the maximum-attended key position, preserving query position. Present summaries as tables or compact heatmaps keyed by layer and head.

Alternative considered: summarize only the globally averaged matrix. That is easier but erases head specialization and makes head comparison impossible.

### 3. Make rollout a separate, documented aggregate

Implement rollout as an explicit derived view with a visible description of layer/head aggregation, normalization, and residual-connection treatment. Keep raw heatmaps available beside it. If the chosen model or tensor shape cannot support the aggregate, show an unavailable message.

Alternative considered: silently include rollout in every heatmap. That would hide methodological assumptions and make users mistake an aggregate for raw model attention.

### 4. Defer clustering until the feature representation is explicit

The first implementation can compare selected heads using summary metrics and raw-pattern similarity. A clustering view should only be enabled once the feature vector, distance metric, and minimum sample conditions are documented. It must not imply that clusters are semantic head types.

Alternative considered: cluster flattened attention matrices immediately. Flattened matrices are strongly affected by prompt length and token positions, so they are a poor default representation.

### 5. Use a capability registry for optional tools

Extend the existing Toolbox registry with display name, package/module, capability description, installation status, and an optional compatibility probe. Run probes only after the user requests them. Keep integration-specific imports inside probe functions.

Initial supported status entries should cover the existing TransformerLens, NNsight, Captum, and UMAP integrations. SAELens, Pyvene, and external visualization packages should be represented as evaluation candidates until a compatible, tested adapter exists.

Alternative considered: import all optional tools at module load time. This would make Streamlit startup and source inspection vulnerable to unrelated package failures.

### 6. Preserve eager-kernel gating

The Attention tab should continue to explain that SDPA and FlashAttention may not return attention matrices. Summary and rollout views are disabled with the raw attention view when the required matrices are absent; non-attention tabs remain usable.

## Risks / Trade-offs

- [Attention matrices scale quadratically with sequence length] -> Keep prompt limits, compute summaries on demand, and avoid duplicating full matrices in session state.
- [Rollout assumptions vary across architectures] -> Show the exact aggregation assumptions and disable unsupported shapes instead of silently adapting.
- [Attention entropy can be misread as importance] -> Define it as concentration over key positions, not causal influence or feature importance.
- [Optional packages conflict with PyTorch or Transformers versions] -> Keep them optional, report compatibility explicitly, and isolate imports and probes.
- [Head comparisons are prompt-dependent] -> Identify prompt and layer context and avoid persistent claims about a head based on one example.

## Migration Plan

No persisted migration is required. Extend the current Toolbox status structure and add derived attention views without changing existing analysis result fields. If an optional integration is incompatible, retain the status row and explanation while leaving the core tabs unchanged. New external packages should be added only as optional extras after a compatibility spike.

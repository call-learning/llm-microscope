## Context

See `proposal.md` and the capability specs for the user-facing behavior. The current application has Gradient x Input attribution, synchronous patching experiments, and layer-wise logistic probes. Patching score curves currently return a single DataFrame per run, while probe results expose accuracy and direction data but do not retain evaluation metadata. Captum is already an optional project extra, so optional attribution methods can be isolated without making the base application depend on it.

## Goals / Non-Goals

**Goals:**

- Preserve the existing intervention and attribution APIs while adding comparable result structures and richer presentation.
- Make attribution sign and method identity explicit.
- Use stratified held-out evaluation when probe data supports it and expose limitations when it does not.
- Store patch metadata with results so matrices cannot combine incompatible experiments.
- Keep expensive attribution, occlusion, and patching work on demand with progress indicators.

**Non-Goals:**

- No claim that any attribution method identifies a complete causal mechanism.
- No automatic concept naming from attribution or probe directions.
- No mandatory Captum dependency and no replacement of the existing Gradient x Input path.
- No full source-layer/target-layer causal interchange matrix in this change; the matrix compares patch layer with target or metric.

## Decisions

### 1. Normalize attribution methods to a common result frame

Represent every method as token position, token label, signed attribution, absolute magnitude, selected output token, and method name. Keep signed values in the canonical frame and derive absolute values only for ranking or a secondary chart.

Alternative considered: let each method return its own DataFrame shape. That would make comparison fragile and would encourage users to compare incompatible columns.

### 2. Use Captum for optional alternative attribution

Implement Integrated Gradients as the initial alternative when Captum is installed. Keep occlusion as a future method because it requires repeated masked forwards and introduces a separate choice about masking token spans. Missing Captum disables only Integrated Gradients and leaves Gradient x Input available.

Alternative considered: implement Integrated Gradients directly. Captum provides established baselines and convergence controls, reducing the chance of a subtly incorrect implementation.

### 3. Display signed attribution with a diverging visualization

Use a diverging bar chart or table styling centered at zero, with positive and negative contributions separately visible. Normalize only for visual comparison and retain the raw score in hover data and tables.

Alternative considered: display absolute importance only. This is simpler but loses the distinction between supporting and opposing evidence.

### 4. Evaluate probes with stratified held-out data

When each class has enough examples, split data using a deterministic stratified split and report train accuracy, held-out accuracy, class counts, and a majority or chance baseline. For small datasets, retain the current training result but label it explicitly as training-only. Estimate uncertainty through repeated stratified splits only when the sample count makes that meaningful.

Alternative considered: always use cross-validation. Cross-validation is more statistically efficient but can produce unstable or misleading estimates for the very small prompt sets the tool encourages.

### 5. Store patch results with experiment metadata

Wrap the existing score DataFrame with metadata for model name, source prompt, target prompt, patch target, attention head when relevant, metric, and analysis settings. Build comparison matrices only from matching metadata and represent missing cells as missing values rather than zero.

Alternative considered: infer compatibility from the DataFrame columns. This cannot detect different prompts or models and would make misleading comparisons easy.

### 6. Keep the existing experiment execution model

Do not automatically run all attribution methods or patch targets. The user explicitly requests a method or comparison, and each run shows progress. This protects the 12 GB GPU use case and keeps the current synchronous Streamlit behavior predictable.

## Risks / Trade-offs

- [Integrated Gradients requires many forward/backward passes] -> Make it opt-in, expose progress, and provide a step-count control with a conservative default.
- [Attribution methods use different scales] -> Identify methods, show raw values, and avoid ranking methods against one another without a normalization explanation.
- [Small probe datasets produce optimistic scores] -> Show train versus held-out labels, baselines, class counts, and a training-only warning.
- [Repeated probe splits can be expensive or unstable] -> Use deterministic seeds, cap repeats, and suppress uncertainty estimates when the dataset is too small.
- [Patching matrices can mix incompatible experiments] -> Require exact metadata compatibility and use explicit missing cells.
- [Captum or model gradients fail for a model] -> Catch optional-method failures, explain the boundary, and keep core views functional.

## Migration Plan

No persisted data migration is required. Add new result dataclasses or metadata fields with local session-state handling, keep existing DataFrame-producing helpers compatible where practical, and render the new views only when their corresponding result exists. If an optional method fails, users can continue with the existing method and patching curve.

# Design

## Context

The application currently performs one Hugging Face forward pass and stores CPU copies of logits, hidden states, and optionally attention matrices. It already has native Plotly views for logit-lens predictions, attention, activations, attribution, probes, and activation patching. `microscope/toolbox.py` reports optional integrations lazily; SAELens, Pyvene, BertViz, and CircuitsVis are candidates without adapters. The existing patching code provides a useful causal execution boundary, but component decomposition and empirical validity analysis are not yet represented as first-class results.

## Goals

- Add useful layer/component views without making optional research packages mandatory.
- Treat validity as calibration, uncertainty, consistency, and sensitivity rather than an asserted internal truth mechanism.
- Reuse existing tokenisation, caching, patching, and Plotly conventions.
- Preserve architecture fallbacks and make every expensive experiment explicit and reproducible.

## Decisions

### 1. Native-first analysis contracts

Native helpers should return small, serialisable DataFrames or dataclasses containing explicit metadata. The app should render these with Plotly and keep optional providers behind adapters. Raw logit lens remains the fallback when tuned-lens artefacts are unavailable.

### 2. Tuned lens as an optional provider

Tuned lens requires a compatible trained translator and model representation. It must not be inferred from a raw logit-lens result. The adapter will report provider/version, model identity, layer count, and artefact compatibility before running.

### 3. Component decomposition before path patching

Implement a common component-result contract and reuse existing hook machinery for residual, attention, MLP, and head targets. Path patching is a later extension of the same contract, not a separate ad-hoc experiment. Contributions must be labelled as direct or cumulative and should use a stable selected-token logit-difference reference.

### 4. Validity experiments use labelled data or explicit variants

Calibration requires user-provided labelled examples. Sensitivity and consistency experiments must preserve generated variants and baseline prompts. No UI will describe confidence, attention, or probe accuracy as truth verification.

### 5. Resource-aware execution

Calibration and perturbation runs execute on demand, report progress, cache compatible results, and enforce the existing prompt/model limits. Attention and gradient-heavy paths retain their current warnings and OOM handling.

## Alternatives considered

- Making TransformerLens or SAELens a base dependency: rejected because model wrappers and artefacts are version- and architecture-sensitive.
- Treating attention maps as validity evidence: rejected because attention is descriptive and not necessarily causal.
- Adding an automatic factuality judge: deferred; an external judge would measure another model's opinion and needs a separate evaluation contract.

## Risks

- Component decompositions may depend on architecture-specific module boundaries; unsupported models must retain residual patching.
- Calibration metrics can be misleading with small or biased datasets; display baselines and limitations.
- Perturbation generation can accidentally alter multiple factors; store exact variants and label the experiment as observational unless intervention is controlled.
- Tuned-lens and SAE artefacts may not match the selected checkpoint; reject mismatches rather than silently substituting artefacts.

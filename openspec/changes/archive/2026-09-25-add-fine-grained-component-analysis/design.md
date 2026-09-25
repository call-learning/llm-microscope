# Design

## Context

The existing runtime discovers decoder layers and common attention/MLP modules. Native intervention code already supports residual, attention-output, MLP-output, and individual-head zero ablation. Hidden states and attention maps are captured by the analysis result, while optional SAE providers are currently represented only in the Toolbox.

## Goals

- Add fine-grained views using explicit, architecture-aware contracts.
- Reuse existing hook cleanup and baseline conventions.
- Keep SAE features optional and reject artefact mismatches.
- Avoid implying that individual neurons or heads are complete concepts.

## Decisions

### 1. Inspect before intervening

Neuron and Q/K/V views should first expose activation tensors for a selected layer/token/head. Ablation controls should run separately and report a stable baseline comparison.

### 2. Use pre-projection hooks for head signals

Individual Q/K/V and head values are architecture-specific. The native adapter should support common Hugging Face modules and return an explicit unavailable result for unsupported module layouts rather than guessing tensor semantics.

### 3. Treat paths as explicit experiments

Path interventions must retain source target names, layer/head/neuron indices, prompts, and selected output metric. A path result is causal evidence for that intervention, not an automatically discovered circuit.

### 4. SAE artefacts are externally versioned

SAE integration must validate model identity, hidden size, layer, feature count, and provider version before decoding features. The core app must not download or substitute artefacts silently.

## Risks

- Fused kernels may not expose Q/K/V tensors; eager/native fallbacks must remain available.
- MLP implementations may fuse projections or use gated intermediate layouts; neuron indexing must be labelled with the supported tensor layout.
- Fine-grained sweeps can be expensive; require explicit selection and cache compatible results.
- Sparse feature explanations may be model- or checkpoint-specific and should be presented as metadata, not ground truth.

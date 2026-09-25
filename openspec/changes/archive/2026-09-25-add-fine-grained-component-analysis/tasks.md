# Tasks

## 1. Contracts and architecture adapters

- [x] 1.1 Define result schemas for neuron activations, Q/K/V signals, path interventions, and SAE features with layer/head/neuron metadata.
- [x] 1.2 Add architecture capability detection for separable MLP intermediates, attention projections, and fused/unsupported layouts.
- [x] 1.3 Add cache identities for component target, token positions, intervention mode, and SAE artefact.

## 2. Neuron and attention signal views

- [x] 2.1 Add MLP neuron activation heatmaps and top-neuron summaries by token position.
- [x] 2.2 Add selected-neuron zero ablation with baseline and selected-output deltas.
- [x] 2.3 Add Q/K/V signal summaries and distinguish them from attention weights.
- [x] 2.4 Add graceful fallbacks and focused tests for fused/unsupported architectures.

## 3. Path interventions

- [x] 3.1 Define supported source-to-target path combinations and retain complete experiment metadata.
- [x] 3.2 Implement one-path intervention experiments with stable unmodified baselines.
- [x] 3.3 Add path comparison charts and interpretation warnings.
- [x] 3.4 Test hook cleanup, target isolation, and failure handling.

## 4. SAE integration

- [x] 4.1 Add lazy SAE provider and artefact status checks to the Toolbox.
- [x] 4.2 Implement exact model/layer/hidden-size compatibility validation.
- [x] 4.3 Add feature activation and feature ablation views when a validated artefact is supplied.
- [x] 4.4 Keep SAE packages and artefacts out of the base dependency/install path.

## 5. Verification and documentation

- [x] 5.1 Document activation, attention signal, ablation, path, and SAE interpretation limits.
- [x] 5.2 Add focused unit tests for schemas, metadata, unsupported layouts, and optional-provider failures.
- [x] 5.3 Run the test suite, environment checks, OpenSpec validation, and diff review.

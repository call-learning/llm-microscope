# Tasks

## 1. Adapter contracts

- [x] 1.1 Define shared adapter status/result schemas for installed, compatible, artefact-ready, adapter-ready, and render-ready states.
- [x] 1.2 Add lazy provider registry and explicit failure isolation.
- [x] 1.3 Add focused tests for missing packages, incompatible models, provider exceptions, and metadata labels.

## 2. Visual providers

- [x] 2.1 Implement BertViz adapter for captured attention tensors and token labels.
- [x] 2.2 Implement CircuitsVis adapter for attention/circuit data and document frontend limitations.
- [x] 2.3 Implement Tuned Lens adapter using validated model/checkpoint artefacts.
- [x] 2.4 Add provider selection and native Plotly fallback views.

## 3. Wrapper and sparse providers

- [x] 3.1 Implement explicit TransformerLens and NNsight trace/intervention adapters.
- [x] 3.2 Implement SAELens artefact/provider adapter and feature metadata validation.
- [x] 3.3 Implement Pyvene intervention adapter for supported model layouts.
- [x] 3.4 Add `saelens` and `pyvene` optional extras without changing base dependencies.

## 4. Toolbox and verification

- [x] 4.1 Update Toolbox statuses, probes, capabilities, and installation guidance.
- [x] 4.2 Verify optional adapter failures leave native views usable.
- [x] 4.3 Run focused tests, environment checks, dependency resolution, and OpenSpec validation.
- [x] 4.4 Document which providers are descriptive, causal, wrapper-based, artefact-based, or rendering-only.

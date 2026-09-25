# fine-grained-components Specification

## Purpose

Expose fine-grained model components and causal paths so users can inspect neurons, attention signals, and sparse features while preserving explicit architecture and interpretation boundaries.

## ADDED Requirements

### Requirement: MLP neuron inspection

The system SHALL display selected-layer MLP neuron activations by token position and SHALL allow a selected neuron to be zero-ablated or compared against the unmodified baseline when the architecture exposes neuron-level intermediate values.

#### Scenario: User selects an MLP neuron

- **WHEN** a compatible layer, token position, and neuron are selected
- **THEN** the interface SHALL show activation values and the selected token context
- **AND** a causal run SHALL report the selected output metric before and after neuron ablation

#### Scenario: Neuron internals are unavailable

- **WHEN** the architecture does not expose a separable MLP intermediate
- **THEN** the interface SHALL report that limitation and preserve layer/module views

### Requirement: Attention signal inspection

The system SHALL provide Q, K, and V signal summaries for selected heads where the model exposes those tensors, with token positions and layer/head identity.

#### Scenario: User inspects a head signal

- **WHEN** a compatible layer and head are selected
- **THEN** the interface SHALL identify whether the displayed signal is Q, K, or V
- **AND** it SHALL distinguish signal magnitude from attention weights

### Requirement: Path intervention

The system SHALL allow a compatible experiment to intervene on a selected source component and report the effect on a selected downstream target or output metric.

#### Scenario: User compares a component path

- **WHEN** a source component, downstream target, prompts, and output metric are selected
- **THEN** the interface SHALL show the unmodified baseline and intervened result
- **AND** it SHALL retain the path metadata and intervention type

### Requirement: Sparse feature inspection

The system SHALL optionally display SAE feature activations and feature ablation effects only when a compatible artefact is available for the exact model representation and layer.

#### Scenario: SAE artefact is incompatible

- **WHEN** the selected SAE does not match the model, layer, or hidden size
- **THEN** the interface SHALL report the mismatch
- **AND** it SHALL not display fabricated or silently substituted feature values

### Requirement: Fine-grained interpretation guidance

Every fine-grained view SHALL state whether it is descriptive or intervention-based and SHALL warn that a high activation or causal effect does not by itself identify a human-readable concept.

#### Scenario: User opens a fine-grained view

- **WHEN** the user opens a neuron, signal, path, or sparse-feature view
- **THEN** the interface SHALL identify the measurement type and intervention status
- **AND** it SHALL show the relevant limitation guidance

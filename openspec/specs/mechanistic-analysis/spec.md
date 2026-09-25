# mechanistic-analysis Specification

## Purpose

Provide component-level and causal validation views that help users distinguish where information is represented from which components causally affect a selected prediction.

## Requirements

### Requirement: Component-level causal comparison

The system SHALL allow compatible models to compare residual, attention-output, MLP-output, and individual-head interventions using a common selected-token effect metric.

#### Scenario: User compares component effects

- **WHEN** the user runs a component intervention sweep
- **THEN** the system SHALL show an effect curve or matrix over layers and component targets
- **AND** the metric SHALL be explicitly identified as a logit, probability, rank, or distribution change
- **AND** the unmodified baseline SHALL be computed once per experiment

### Requirement: Probe and intervention comparison

The system SHALL provide a comparison between representation decodability and intervention effect when both a probe and compatible intervention results exist.

#### Scenario: Decodable but not causally influential

- **WHEN** a property is decodable at a layer but its intervention has little selected-output effect
- **THEN** the interface SHALL preserve both measurements
- **AND** it SHALL warn that decodability does not establish that the model uses the property

### Requirement: Controlled causal experiment metadata

Every mechanistic result SHALL retain source and target prompts, layer, component, head where relevant, intervention type, baseline, and selected output metric.

#### Scenario: User reviews an intervention result

- **WHEN** a mechanistic experiment completes
- **THEN** the result SHALL expose the experiment inputs and selected metric alongside the plotted values

### Requirement: Optional provider isolation

Optional tracing or sparse-feature providers SHALL be invoked only through explicit adapters and SHALL not prevent native residual and patching views from loading or operating.

#### Scenario: Optional provider fails

- **WHEN** an optional provider is missing, incompatible, or raises an execution error
- **THEN** the native residual and patching views SHALL remain available
- **AND** the failure SHALL be reported as an unavailable optional view

### Requirement: Fine-grained causal targets

The system SHALL extend compatible mechanistic experiments with neuron and path targets while retaining residual, module, and head targets.

#### Scenario: User selects a fine-grained causal target

- **WHEN** the selected architecture exposes the requested target
- **THEN** the system SHALL report the same baseline, intervention, and selected-output metrics used by existing component analysis
- **AND** it SHALL identify the target layer, neuron/head, and path metadata

#### Scenario: Fine-grained target is unsupported

- **WHEN** the architecture cannot expose the requested target
- **THEN** the system SHALL explain the unsupported boundary and preserve compatible mechanistic targets

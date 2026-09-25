# mechanistic-analysis Specification Delta

## ADDED Requirements

### Requirement: Fine-grained causal targets

The system SHALL extend compatible mechanistic experiments with neuron and path targets while retaining residual, module, and head targets.

#### Scenario: User selects a fine-grained causal target

- **WHEN** the selected architecture exposes the requested target
- **THEN** the system SHALL report the same baseline, intervention, and selected-output metrics used by existing component analysis
- **AND** it SHALL identify the target layer, neuron/head, and path metadata

#### Scenario: Fine-grained target is unsupported

- **WHEN** the architecture cannot expose the requested target
- **THEN** the system SHALL explain the unsupported boundary and preserve compatible mechanistic targets

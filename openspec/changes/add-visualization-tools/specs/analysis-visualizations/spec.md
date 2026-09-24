# Spec Delta

## ADDED Requirements

### Requirement: Additional visualisation providers preserve core views

Any optional provider added for interactive attention, circuit, activation, or sparse-feature visualisation SHALL be additive and SHALL NOT replace or disable visualisations that can be computed by the core workbench.

#### Scenario: Optional provider is unavailable

- **WHEN** the user requests a visualisation whose optional provider is missing or incompatible
- **THEN** the interface SHALL explain why that provider cannot be used
- **AND** SHALL preserve any equivalent core visualisation that remains computable

### Requirement: Optional visualisations identify their data and assumptions

An optional visualisation SHALL identify the representation, token positions, layers, heads, or features it displays when those dimensions apply, and SHALL explain important approximation, aggregation, or model-conversion assumptions.

#### Scenario: User opens an optional attention or circuit visualisation

- **WHEN** the visualisation is displayed
- **THEN** the interface SHALL identify the relevant tokens and model dimensions or explain why they are unavailable
- **AND** SHALL state whether the view is descriptive, approximate, aggregated, or causal

### Requirement: Optional visualisation failures are recoverable

The system SHALL isolate errors from optional visualisation providers so that a provider failure does not terminate the application or invalidate the current core analysis.

#### Scenario: Provider raises an integration error

- **WHEN** an optional visualisation provider fails while preparing or rendering a view
- **THEN** the interface SHALL show an actionable error or unavailable-state message
- **AND** the user SHALL be able to continue using other analysis tabs and visualisations

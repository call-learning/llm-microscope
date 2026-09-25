# advanced-tool-integrations Specification Delta

## ADDED Requirements

### Requirement: Mechanistic and sparse-feature adapter status

The Toolbox SHALL distinguish package installation, selected-model compatibility, artefact availability, and application-adapter availability for tuned-lens, TransformerLens, and SAELens integrations.

#### Scenario: Provider is installed without a compatible artefact

- **WHEN** an optional provider is installed but its model wrapper or tuned-lens/SAE artefact is unavailable
- **THEN** the Toolbox SHALL report the missing compatibility boundary
- **AND** native Hugging Face analysis SHALL remain available

### Requirement: Optional visual output provenance

An optional mechanistic or sparse-feature view SHALL identify its provider, model representation, artefact/version, and interpretation assumptions before displaying results.

#### Scenario: Optional provider produces a view

- **WHEN** an optional mechanistic or sparse-feature adapter returns a result
- **THEN** the interface SHALL display the provider, model representation, artefact or version, and assumptions
- **AND** it SHALL retain the native view as an alternative where available

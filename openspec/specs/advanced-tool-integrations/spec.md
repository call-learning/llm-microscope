# advanced-tool-integrations Specification

## Purpose

Make optional interpretability integrations discoverable and safe by reporting their installation and model compatibility without allowing them to block the core Hugging Face-based workbench.

## Requirements

### Requirement: Optional tool capability status

The system SHALL report whether each optional integration is installed and SHALL distinguish installation status from compatibility with the currently selected model.

#### Scenario: User opens the Toolbox

- **WHEN** the Toolbox is displayed
- **THEN** the system SHALL show installation status for supported optional tools
- **AND** compatibility SHALL be reported separately when it has been checked

### Requirement: Optional tool compatibility checks

The system SHALL allow an explicit compatibility check for supported integrations and SHALL report success, unsupported model behavior, or an actionable failure without crashing the main application.

#### Scenario: Compatibility check fails

- **WHEN** an optional tool cannot load the selected model
- **THEN** the Toolbox SHALL report the failure and likely compatibility boundary
- **AND** the core analysis views SHALL remain available

### Requirement: Optional dependency isolation

The system SHALL keep optional integrations unavailable rather than silently substituting a different analysis method when dependencies are missing.

#### Scenario: Optional package is not installed

- **WHEN** a user selects a feature backed by an absent optional package
- **THEN** the interface SHALL explain what package enables the feature
- **AND** it SHALL not present an empty or fabricated result

### Requirement: Integration interpretation guidance

The Toolbox SHALL explain the purpose and scope of each supported integration, including whether it provides tracing, interventions, attribution, sparse features, or visualization only.

#### Scenario: User evaluates an advanced tool

- **WHEN** the user views an integration's status
- **THEN** the interface SHALL describe what kind of analysis the tool adds
- **AND** it SHALL identify relevant model, device, or memory limitations

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

### Requirement: SAE artefact compatibility status

The Toolbox SHALL report SAE provider installation, adapter availability, model compatibility, layer coverage, and artefact availability separately.

#### Scenario: User checks sparse-feature support

- **WHEN** the user checks SAE support for the selected model
- **THEN** the Toolbox SHALL identify whether a compatible artefact is available
- **AND** a missing or incompatible artefact SHALL not disable native analysis

### Requirement: Adapter readiness status

The Toolbox SHALL report package installation, model compatibility, artefact readiness, adapter readiness, and rendering readiness as separate statuses for each optional integration.

#### Scenario: Provider is installed but incompatible

- **WHEN** an optional package imports but cannot handle the selected model or tensor format
- **THEN** the Toolbox SHALL report the compatibility failure
- **AND** native analysis SHALL remain available

### Requirement: Explicit provider adapters

Optional providers SHALL be invoked only through named adapters that return labelled, view-ready data or an actionable unavailable result.

#### Scenario: Provider adapter fails

- **WHEN** an adapter raises an import, model, artefact, or rendering error
- **THEN** the Toolbox SHALL identify the failing provider and boundary
- **AND** the application SHALL not fabricate a result or disable unrelated views

### Requirement: Optional package installation guidance

The Toolbox SHALL identify the project extra or installation command required for an unavailable optional provider.

#### Scenario: SAELens or Pyvene is missing

- **WHEN** a user checks an uninstalled provider
- **THEN** the Toolbox SHALL show the corresponding optional extra
- **AND** it SHALL preserve all native views

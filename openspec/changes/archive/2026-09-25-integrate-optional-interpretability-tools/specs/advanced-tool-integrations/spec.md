# advanced-tool-integrations Specification Delta

## ADDED Requirements

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

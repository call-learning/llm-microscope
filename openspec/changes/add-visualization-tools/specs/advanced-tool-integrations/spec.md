# Spec Delta

## ADDED Requirements

### Requirement: Visualisation integration lifecycle is explicit

The Toolbox SHALL distinguish whether an optional visualisation integration is installed, compatible with the selected model, and connected to an application feature. A tool SHALL NOT be presented as usable solely because its package is installed.

#### Scenario: Candidate tool is installed but not integrated

- **WHEN** the user views the Toolbox and an optional visualisation package is installed without an application adapter
- **THEN** the Toolbox SHALL report it as installed but not integrated
- **AND** SHALL explain that no application visualisation is currently available through that tool

#### Scenario: Candidate tool is unavailable

- **WHEN** the user views the Toolbox and an optional visualisation package is not installed
- **THEN** the Toolbox SHALL identify the missing package or integration
- **AND** SHALL keep the core visualisation views available

### Requirement: Visualisation compatibility checks are isolated

The system SHALL allow an explicit compatibility check for an optional visualisation integration without making the core workbench depend on that integration.

#### Scenario: Compatibility check fails

- **WHEN** an optional visualisation integration cannot load the selected model or cannot produce its required representation
- **THEN** the Toolbox SHALL report the failure and a useful compatibility boundary
- **AND** the core analysis and visualisation views SHALL remain usable

### Requirement: Optional visualisation limitations are disclosed

The Toolbox SHALL describe the kind of output an optional visualisation integration provides and SHALL identify relevant model, frontend, memory, or dependency limitations before the user relies on it.

#### Scenario: User inspects an optional visualisation tool

- **WHEN** the user views the status of an optional visualisation integration
- **THEN** the interface SHALL identify whether the tool provides attention, circuit, activation, or sparse-feature visualisation
- **AND** SHALL state known limitations and whether an application adapter is enabled

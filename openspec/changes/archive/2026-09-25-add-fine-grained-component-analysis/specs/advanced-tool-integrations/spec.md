# advanced-tool-integrations Specification Delta

## ADDED Requirements

### Requirement: SAE artefact compatibility status

The Toolbox SHALL report SAE provider installation, adapter availability, model compatibility, layer coverage, and artefact availability separately.

#### Scenario: User checks sparse-feature support

- **WHEN** the user checks SAE support for the selected model
- **THEN** the Toolbox SHALL identify whether a compatible artefact is available
- **AND** a missing or incompatible artefact SHALL not disable native analysis

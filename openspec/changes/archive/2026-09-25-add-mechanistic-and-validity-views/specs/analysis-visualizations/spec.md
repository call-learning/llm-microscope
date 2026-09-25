# analysis-visualizations Specification Delta

## ADDED Requirements

### Requirement: Tuned layer prediction view

The system SHALL optionally provide a tuned-lens view that translates intermediate residual states before decoding them, alongside the existing raw logit lens.

#### Scenario: User compares raw and tuned lens

- **WHEN** a compatible tuned-lens provider and artefact are available
- **THEN** the system SHALL show layer-wise predictions for both the raw and tuned methods
- **AND** the view SHALL identify the selected method, layer, token, and model artefact
- **AND** it SHALL explain that neither method is a literal record of the model's completed reasoning

#### Scenario: Tuned lens is unavailable

- **WHEN** no compatible tuned-lens artefact or provider is available
- **THEN** the existing raw logit-lens view SHALL remain usable
- **AND** the interface SHALL report the specific compatibility or installation boundary

### Requirement: Component contribution visualisation

The system SHALL provide a view that compares the contribution of embeddings, attention outputs, MLP outputs, and residual updates to a selected output logit when the model architecture exposes the required values.

#### Scenario: User inspects a selected output token

- **WHEN** the user selects an output token and runs component analysis
- **THEN** the system SHALL display signed component contributions by layer
- **AND** it SHALL identify whether values are direct, projected, or cumulative estimates

#### Scenario: Architecture does not expose components

- **WHEN** the selected architecture cannot provide a component decomposition
- **THEN** the interface SHALL preserve the existing layer and activation views
- **AND** it SHALL explain why component analysis is unavailable rather than showing fabricated zeros

### Requirement: Visual interpretation boundaries

Each layer visualisation SHALL identify whether it is descriptive, correlational, or intervention-based and SHALL expose the selected prompt, token, layer range, and method assumptions.

#### Scenario: User opens a layer visualisation

- **WHEN** the user opens a raw lens, tuned lens, or component view
- **THEN** the interface SHALL identify the method and its interpretation category
- **AND** it SHALL show the prompt, selected token, and relevant layer range

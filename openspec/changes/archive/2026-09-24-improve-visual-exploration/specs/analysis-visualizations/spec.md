## Purpose

Provide connected, interpretable visual views that help users understand how a decoder-only language model develops predictions and representations across token positions and layers.

## ADDED Requirements

### Requirement: Token-by-layer exploration

The system SHALL provide a visual view combining token positions with layer-by-layer model measurements, including the selected token's probability or logit, rank among candidate next tokens, and at least one representation measure such as hidden-state norm.

#### Scenario: User inspects prediction development

- **WHEN** the user has run an analysis and opens the token-by-layer view
- **THEN** the system SHALL display the prompt's token positions and model layers on identifiable axes
- **AND** the view SHALL allow the user to distinguish changes across layers from changes across token positions

#### Scenario: User selects a candidate token

- **WHEN** the user selects a candidate output token from the available prediction results
- **THEN** the view SHALL update to show that token's probability, logit, or rank at each available layer
- **AND** the selected token SHALL be clearly identified in the chart

### Requirement: Next-token uncertainty visualisation

The system SHALL present the final next-token distribution as a visual ranking of candidate tokens and SHALL provide an uncertainty metric such as entropy for the relevant prediction positions or layers.

#### Scenario: User compares candidate predictions

- **WHEN** the user opens the next-token prediction view after analysis
- **THEN** the system SHALL show candidate tokens ordered by probability
- **AND** each candidate SHALL have a readable probability value

#### Scenario: User examines uncertainty

- **WHEN** the user selects an uncertainty view
- **THEN** the system SHALL show how entropy changes over the requested token positions or layers
- **AND** the view SHALL explain that higher entropy indicates a less concentrated distribution, not necessarily an incorrect prediction

### Requirement: Representation comparison views

The system SHALL provide at least one visual representation comparison beyond a single scalar curve, including a token-by-layer heatmap, a pairwise similarity or distance view, or a representation trajectory across layers.

#### Scenario: User inspects representation changes

- **WHEN** the user selects a layer range or representation view
- **THEN** the system SHALL display token labels or positions and identify the corresponding layer or layers
- **AND** the visualisation SHALL include enough context to relate a pattern to the original prompt

#### Scenario: Projection cannot be computed

- **WHEN** the selected representation view requires more tokens or dimensions than the available analysis provides
- **THEN** the system SHALL show a clear explanation and retain any views that can still be computed

### Requirement: Interpretation guidance

Each new visualisation SHALL provide concise definitions of its displayed metrics and a warning where the metric is correlational, approximate, or otherwise insufficient as a standalone explanation.

#### Scenario: User opens a new visualisation

- **WHEN** the user opens the visualisation without prior use of the tool
- **THEN** the interface SHALL explain what the chart displays, what question it can help answer, and any important interpretation limitation

### Requirement: On-demand visual analysis

The system SHALL compute expensive visualisation data on user request, reuse compatible analysis results where possible, and show progress while additional model inference is running.

#### Scenario: User requests an expensive view

- **WHEN** a visualisation requires additional model inference or a costly transformation
- **THEN** the interface SHALL indicate that work is in progress
- **AND** the interface SHALL remain usable after the result is displayed

#### Scenario: Existing analysis is sufficient

- **WHEN** a requested view can be derived from the current analysis result without new inference
- **THEN** the system SHALL derive the view from that result rather than rerunning the model unnecessarily

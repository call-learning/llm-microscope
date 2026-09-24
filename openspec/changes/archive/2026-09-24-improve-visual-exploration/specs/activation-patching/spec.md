## ADDED Requirements

### Requirement: Patching effect matrix view

The system SHALL provide a matrix or heatmap representation of patching effects that makes the effect of patch layer and patch target directly comparable, while retaining the existing per-layer score curve.

#### Scenario: User views patch effects across layers

- **WHEN** the user completes a patching score-curve experiment
- **THEN** the system SHALL offer a heatmap or equivalent matrix with layers identifiable on an axis and an effect metric represented by color or value
- **AND** the view SHALL identify whether the displayed effect is a logit change or probability change

#### Scenario: User compares patch targets

- **WHEN** results are available for more than one supported patch target
- **THEN** the system SHALL allow the user to distinguish residual, attention-output, and MLP-output effects
- **AND** unsupported targets SHALL remain clearly identified as unavailable rather than appearing as zero effects

### Requirement: Patching visual interpretation

The patching results SHALL explain the baseline used for comparison and SHALL clarify that a changed output is evidence of causal influence of the intervention, not automatically a human-readable concept attribution.

#### Scenario: User reads a patching result

- **WHEN** the patching curve or matrix is displayed
- **THEN** the interface SHALL identify the source prompt, target prompt, selected patch target, and comparison baseline
- **AND** the interface SHALL display a concise causal-interpretation warning

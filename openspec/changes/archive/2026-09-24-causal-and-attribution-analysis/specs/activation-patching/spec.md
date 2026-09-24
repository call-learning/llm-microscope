## ADDED Requirements

### Requirement: Patching effect matrix

The system SHALL provide a matrix or heatmap representation of compatible patching results with patch layer on one axis and supported patch target or effect metric on the other, while retaining the existing score curve.

#### Scenario: User compares patch targets

- **WHEN** the user has completed compatible score curves for multiple patch targets
- **THEN** the system SHALL display their effects in a directly comparable matrix or heatmap
- **AND** the matrix SHALL identify the patch layer and effect metric

#### Scenario: Only one patch target is available

- **WHEN** the user has completed a score curve for only one patch target
- **THEN** the system SHALL display a useful single-target matrix or equivalent view
- **AND** it SHALL explain that cross-target comparison requires additional runs

### Requirement: Patching experiment compatibility

The system SHALL compare patching runs only when their model, source prompt, target prompt, and metric are compatible, and SHALL not represent unavailable or unrun targets as zero effects.

#### Scenario: Incompatible runs are present

- **WHEN** stored patching results use different prompts, models, or metrics
- **THEN** the system SHALL keep them separate or request a compatible selection
- **AND** it SHALL explain why they cannot be combined

### Requirement: Causal interpretation guidance

The system SHALL identify the baseline, source prompt, target prompt, and intervention target in patching views and SHALL explain that an output change is evidence of intervention influence rather than automatic concept identification.

#### Scenario: User reads a patching matrix

- **WHEN** the matrix or curve is displayed
- **THEN** the interface SHALL show the experiment context and baseline
- **AND** it SHALL show the causal-interpretation warning

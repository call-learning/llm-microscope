# probe-diagnostics Specification

## Purpose

Provide reliability context for layer-wise linear probes so users can distinguish representation decodability from probe overfitting, class imbalance, or chance-level performance.

## Requirements

### Requirement: Probe evaluation diagnostics

The system SHALL report train and held-out evaluation performance for each probed layer when the dataset permits a held-out evaluation, and SHALL show a chance or majority-class baseline.

#### Scenario: User evaluates a probe

- **WHEN** the user fits a probe with enough labeled examples for a held-out evaluation
- **THEN** the system SHALL show per-layer training performance and held-out performance
- **AND** the chart SHALL include an appropriate baseline reference

#### Scenario: Dataset is too small for a split

- **WHEN** the provided examples cannot support a meaningful held-out evaluation
- **THEN** the system SHALL explain that the reported result is training-only or otherwise limited
- **AND** it SHALL not label training accuracy as generalization performance

### Requirement: Probe dataset warnings

The system SHALL warn when labels are imbalanced, when there are too few examples for a class, or when the result may be dominated by a prompt-format shortcut.

#### Scenario: Imbalanced labels are provided

- **WHEN** one class substantially outnumbers the other
- **THEN** the system SHALL display an imbalance warning
- **AND** it SHALL show the class counts used by the probe

### Requirement: Probe uncertainty context

The system SHALL provide uncertainty context for probe performance, such as confidence intervals or repeated-split variation, when the selected evaluation method supports it.

#### Scenario: User views probe uncertainty

- **WHEN** repeated evaluation or confidence estimation is available
- **THEN** the system SHALL display the uncertainty alongside the per-layer score
- **AND** the interface SHALL explain what the uncertainty measure represents

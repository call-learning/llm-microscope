# validity-analysis Specification

## Purpose

Help users assess confidence, calibration, consistency, and sensitivity of model outputs without implying that a language model has a single internal truth or validity checker.

## ADDED Requirements

### Requirement: Confidence and uncertainty view

The system SHALL display selected-output confidence measures including probability, entropy, top-token margin, surprisal, and sequence likelihood where computable.

#### Scenario: User inspects a prediction

- **WHEN** the user opens the validity view for an analysed prompt
- **THEN** the system SHALL show the selected metric definitions and the relevant token or answer scope
- **AND** it SHALL explain that confidence is not factual correctness

### Requirement: Calibration evaluation

The system SHALL accept a labelled evaluation set and report accuracy, reliability diagrams, expected calibration error, and Brier score when enough examples are available.

#### Scenario: Evaluation set is too small or imbalanced

- **WHEN** calibration estimates are unreliable because of sample size or class distribution
- **THEN** the interface SHALL show the available result with an uncertainty or limitation warning
- **AND** it SHALL identify the baseline and evaluation split used

### Requirement: Perturbation sensitivity

The system SHALL support controlled prompt variants, including paraphrase, irrelevant-context, evidence replacement, masking, and contradiction variants when the selected experiment supports them.

#### Scenario: User compares variants

- **WHEN** the user runs a perturbation experiment
- **THEN** the system SHALL compare answer probability, answer identity, uncertainty, and representation similarity across variants
- **AND** it SHALL retain the exact variant construction and baseline prompt

### Requirement: Consistency analysis

The system SHALL support repeated or variant-based answer agreement measurements and SHALL distinguish agreement from correctness.

#### Scenario: User runs repeated variants

- **WHEN** repeated or variant prompts produce answers
- **THEN** the system SHALL report agreement and identify the compared variants
- **AND** it SHALL state that agreement does not establish correctness

### Requirement: Validity interpretation guidance

The interface SHALL state that calibration and consistency are empirical reliability signals, while attribution, attention, probes, and activation interventions answer different questions and do not independently establish factual validity.

#### Scenario: User opens validity analysis

- **WHEN** the validity workspace is displayed
- **THEN** the interface SHALL show the distinction between confidence, calibration, consistency, representation, and causal influence

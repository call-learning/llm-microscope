# linear-probing Specification

## Purpose

Provide an observational linear-probe view that locates the decoder layer at which a binary property of the prompt becomes linearly decodable from the final-token hidden state, complementing the causal patching views.

## Requirements

### Requirement: Per-layer binary linear probe

The system SHALL fit a binary linear classifier (L2-regularized logistic regression) on the final-token hidden state at each decoder layer, using user-supplied labels, and report per-layer accuracy so the user can see at which layer the property becomes decodable.

#### Scenario: Probe accuracy reported per layer

- **WHEN** the user supplies a labeled training set and runs the probe
- **THEN** the system SHALL fit one binary classifier per decoder layer on the final-token hidden state
- **AND** report the per-layer accuracy
- **AND** present the result in a form that shows where accuracy rises across layers

### Requirement: L2 penalty is configurable

The system SHALL use L2 (ridge) regularization for the probe classifier and SHALL expose the penalty strength to the user so it can be adjusted.

#### Scenario: Penalty strength is adjustable

- **WHEN** the user changes the penalty-strength control and re-runs the probe
- **THEN** the fitted classifier SHALL use the selected L2 penalty

### Requirement: Contrastive (concept) label mode

The system SHALL provide a concept label mode in which the user supplies a positive prompt template, a negative prompt template, and a comma-separated list of fill values. The system SHALL fill each template with each fill value to produce a balanced set of positive and negative prompts, and label each generated prompt accordingly.

#### Scenario: Concept templates produce a balanced labeled set

- **WHEN** the user provides a positive template, a negative template, and N fill values in concept mode
- **THEN** the system SHALL generate N positive prompts (positive template filled) labeled 1 and N negative prompts (negative template filled) labeled 0
- **AND** use that labeled set as probe training data

### Requirement: Quick (surface) label mode

The system SHALL provide a quick label mode in which the user supplies a property substring and the system labels each generated prompt by whether it contains the substring.

#### Scenario: Substring label mode

- **WHEN** the user provides a property substring and a set of prompts in quick mode
- **THEN** the system SHALL label each prompt 1 if it contains the substring and 0 otherwise
- **AND** use those labeled prompts as probe training data

### Requirement: Learned direction is reported

The system SHALL, for the fitted probe, report the learned weight vector (probe direction) in addition to per-layer accuracy, so a single probe fit yields both the accuracy curve and a reusable direction.

#### Scenario: Direction reported alongside accuracy

- **WHEN** the user runs the probe
- **THEN** the system SHALL report the classifier's weight vector for the selected/fitted layer in addition to the accuracy

### Requirement: Shared fitting core across label modes

Both the concept and quick label modes SHALL produce a labeled training set of the same shape (prompts with binary labels) and SHALL be fed to a single shared fitting routine, so that the only difference between modes is how labels are generated.

#### Scenario: Both modes use one fitting routine

- **WHEN** the user runs the probe in either concept mode or quick mode
- **THEN** the label generation differs but the subsequent fitting and reporting SHALL use the same routine

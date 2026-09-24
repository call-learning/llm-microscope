# attribution-analysis Specification

## Purpose

Provide comparable token-attribution methods that help users inspect local evidence for a selected output token without presenting any single method as a complete explanation.

## Requirements

### Requirement: Multiple attribution methods

The system SHALL offer Gradient x Input and at least one alternative method, such as Integrated Gradients or occlusion, for a selected prompt and output token.

#### Scenario: User compares attribution methods

- **WHEN** the user selects a prompt, output token, and two available attribution methods
- **THEN** the system SHALL display a token-level result for each method
- **AND** the interface SHALL identify the method used for every result

#### Scenario: Optional attribution dependency is unavailable

- **WHEN** an attribution method requires an optional library that is not installed
- **THEN** the system SHALL mark that method unavailable with an installation explanation
- **AND** Gradient x Input SHALL remain usable if its requirements are available

### Requirement: Signed attribution display

The system SHALL preserve the sign of attribution scores and SHALL distinguish tokens that support the selected output token from tokens that oppose it.

#### Scenario: User inspects positive and negative contributions

- **WHEN** an attribution result contains both positive and negative contributions
- **THEN** the visualization SHALL render them with distinguishable direction or color
- **AND** the table SHALL expose the signed values rather than only absolute magnitude

### Requirement: Attribution limitations

The system SHALL explain that attribution results are local to the selected output token and method, and SHALL warn that agreement between methods is not proof of a complete causal explanation.

#### Scenario: User opens attribution results

- **WHEN** attribution results are displayed
- **THEN** the interface SHALL identify the selected output token and method
- **AND** it SHALL show the local-sensitivity and method-dependence warning

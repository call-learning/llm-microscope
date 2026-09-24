# attention-analysis Specification

## Purpose

Provide interpretable attention summaries and comparisons across layers and heads while clearly separating descriptive attention patterns from causal explanations.

## Requirements

### Requirement: Attention heatmap usability

The system SHALL display attention heatmaps with token positions, readable token labels or hover text, selected layer and head, and a clear indication of query and key axes.

#### Scenario: User inspects a head

- **WHEN** the user selects an available layer and attention head
- **THEN** the heatmap SHALL identify the selected layer and head
- **AND** hovering or inspecting a cell SHALL identify the query token, key token, and attention value

### Requirement: Attention head summaries

The system SHALL provide summary metrics for selected or comparable heads, including attention entropy and the position receiving the maximum attention where those metrics are computable.

#### Scenario: User compares heads

- **WHEN** the user requests a head summary
- **THEN** the system SHALL show comparable metrics for the selected heads
- **AND** the summary SHALL preserve layer and head identity

### Requirement: Attention aggregate views

The system SHALL provide an explicitly labeled aggregate attention view, such as attention rollout, only when the aggregation method and normalization assumptions are documented in the interface.

#### Scenario: User opens an aggregate view

- **WHEN** the user selects an attention rollout or aggregate view
- **THEN** the system SHALL describe how layers, heads, and residual contributions were combined
- **AND** the interface SHALL warn that the aggregate is descriptive and not automatically causal

### Requirement: Attention availability handling

The system SHALL explain when attention matrices are unavailable because of the selected attention kernel and SHALL preserve access to non-attention analyses.

#### Scenario: Optimized kernel does not expose attention

- **WHEN** the selected model execution mode does not return attention matrices
- **THEN** the Attention tab SHALL explain how to enable a compatible mode
- **AND** the other analysis tabs SHALL remain usable

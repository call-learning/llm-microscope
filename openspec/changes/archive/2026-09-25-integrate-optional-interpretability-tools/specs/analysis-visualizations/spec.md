# analysis-visualizations Specification Delta

## ADDED Requirements

### Requirement: Optional provider rendering

The system SHALL allow a compatible optional provider to render a labelled attention, circuit, tracing, or sparse-feature view while preserving the equivalent native view as a fallback.

#### Scenario: BertViz or CircuitsVis renders successfully

- **WHEN** the provider is installed, compatible, and explicitly selected
- **THEN** the interface SHALL identify the provider, model, tokens, and represented layers/heads/features
- **AND** the native Plotly view SHALL remain available

#### Scenario: Browser-oriented rendering is unavailable

- **WHEN** a provider cannot render inside the current Streamlit/browser context
- **THEN** the interface SHALL report the rendering boundary
- **AND** it SHALL show the native Plotly equivalent when available

# activation-patching Specification

## Purpose

Provide causal intervention views over a decoder-only model's residual stream and module outputs so a user can locate which layer or component is responsible for a prediction change between two prompts.

## Requirements

### Requirement: Residual-stream patching

The system SHALL allow the user to copy the final-token hidden state at a selected layer from a source prompt into the same layer of a target prompt's forward pass and report the resulting next-token prediction. This target SHALL work for any supported decoder architecture.

#### Scenario: Single-layer residual patch changes the prediction

- **WHEN** the user selects the residual target, a source prompt, a target prompt, and a layer, and runs the patch
- **THEN** the system reports the target's top next-token probabilities both without and with the source final-token state injected at that layer
- **AND** the patched result reflects the copied state at the selected layer

#### Scenario: Residual patching is available on any supported architecture

- **WHEN** the loaded model does not follow the common per-layer attention/MLP submodule convention
- **THEN** residual-stream patching at a selected layer SHALL still be available and produce a result

### Requirement: Module-level patching targets

The system SHALL allow the user to patch, at a selected layer, either the attention block's output or the MLP's output in addition to the residual stream. For attention outputs the patch SHALL be exposed at the query-head level. On architectures that do not follow the common per-layer attention/MLP submodule convention, the system SHALL surface a clear message that the module target is unavailable for that architecture while keeping the residual target usable.

#### Scenario: Attention-output patch is exposed per query head

- **WHEN** the user selects the attention-output target on a model with group-query attention
- **THEN** the selectable patch position SHALL correspond to a query head, not a key/value head

#### Scenario: Module target unavailable on unsupported architecture

- **WHEN** the user selects the attention-output or MLP target on a model that does not expose the expected per-layer submodules
- **THEN** the system SHALL show a clear message that the module target is unavailable for this architecture
- **AND** the residual-stream target SHALL remain selectable and functional

### Requirement: Patching score curve

The system SHALL, for a source→target prompt pair, compute how the target's next-token prediction changes when the final-token state is patched at every layer and present the result as a curve over layers. The curve SHALL report, for each layer, the effect of the patch (for example the change in the top-token probability or logit) relative to the un-patched target. The sweep SHALL be computed with a deduplicated forward-pass loop in which the source hidden states are captured once, the un-patched target is run once, and only one patched forward is executed per layer.

#### Scenario: Patching score curve across all layers

- **WHEN** the user provides a source prompt, a target prompt, and a patch target, and requests the score curve
- **THEN** the system SHALL return one entry per decoder layer describing the effect of patching that layer
- **AND** the result SHALL be presentable as a curve over layers

#### Scenario: Score curve uses a deduplicated forward loop

- **WHEN** the score curve is computed for a model with N decoder layers
- **THEN** the system SHALL capture the source hidden states in a single source forward, compute the un-patched target baseline in a single target forward, and run exactly one patched forward per layer (N patched forwards total)

#### Scenario: Baseline is not recomputed per layer

- **WHEN** the score curve is computed
- **THEN** the un-patched target baseline SHALL be computed once, not once per layer

### Requirement: Mean ablation

The system SHALL allow the user to ablate a selected target (residual stream, attention output, or MLP output) at a layer by replacing its value with either zero or the mean of the source-run value, without requiring a clean target-matched source state, and report the resulting next-token prediction.

#### Scenario: Zero ablation at a layer

- **WHEN** the user selects a target and layer and chooses zero ablation
- **THEN** the system SHALL set the target's final-token contribution at that layer to zero during the target forward
- **AND** report the resulting next-token prediction compared to the un-ablated baseline

#### Scenario: Mean ablation at a layer

- **WHEN** the user selects a target and layer and chooses mean ablation
- **THEN** the system SHALL replace the target's final-token value at that layer with the mean of the source-run value
- **AND** report the resulting next-token prediction compared to the un-ablated baseline

### Requirement: Synchronous on-demand execution

The score curve and ablation experiments SHALL run synchronously on user request behind a progress indicator, consistent with the existing on-demand analysis pattern, and SHALL not block other tabs.

#### Scenario: Experiment runs behind a progress indicator

- **WHEN** the user requests a score curve or ablation
- **THEN** the system SHALL display a progress indicator while the forwards run
- **AND** present the result when complete

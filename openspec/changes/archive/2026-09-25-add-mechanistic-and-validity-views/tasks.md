# Tasks

## 1. Analysis contracts and caching

- [x] 1.1 Define result dataclasses/DataFrame schemas for tuned-lens metrics, component contributions, validity metrics, calibration runs, and perturbation runs.
- [x] 1.2 Add cache keys containing model identity, prompt/data identity, layer range, target token, method, and provider artefact.
- [x] 1.3 Add focused unit tests for empty inputs, token alignment, unsupported architectures, and metadata preservation.

## 2. Tuned-lens and layer views

- [x] 2.1 Add a lazily loaded tuned-lens adapter with explicit model/artefact compatibility checks and graceful fallback.
- [x] 2.2 Add raw-versus-tuned lens charts, rank/KL comparisons, and interpretation guidance to the layer visualisation page.
- [x] 2.3 Verify memory use and behaviour when the optional provider is absent.

## 3. Mechanistic decomposition and causal validation

- [x] 3.1 Extend native hook utilities to capture signed component contributions for residual, attention, MLP, and supported head targets.
- [x] 3.2 Add component curves/matrices with stable baseline and selected-token logit/probability metrics.
- [x] 3.3 Add probe-versus-intervention comparison using compatible stored results, with decodability/use warnings.
- [x] 3.4 Add tests that verify hook cleanup, one baseline execution, metadata, and architecture fallback.

## 4. Validity and confidence workspace

- [x] 4.1 Add confidence and uncertainty metrics for token and sequence predictions.
- [x] 4.2 Add labelled calibration evaluation with reliability diagram, ECE, Brier score, accuracy, and baseline reporting.
- [x] 4.3 Add controlled variant generation/import and perturbation comparison for masking, contradiction, irrelevant context, and paraphrase cases.
- [x] 4.4 Add repeated/variant answer agreement and representation-similarity summaries.
- [x] 4.5 Add small-data, imbalance, and correctness-versus-confidence warnings.

## 5. Optional integrations and documentation

- [x] 5.1 Extend Toolbox metadata and probes for tuned-lens, TransformerLens, and SAELens provider, artefact, and adapter states.
- [x] 5.2 Keep optional dependencies in dedicated extras and update the lockfile only for an accepted adapter.
- [x] 5.3 Document the difference between representation, causal influence, confidence, calibration, consistency, and factual correctness.

## 6. Verification

- [x] 6.1 Run focused unit tests for analysis helpers and validity metrics.
- [x] 6.2 Run the application smoke/environment checks with optional packages both absent and installed where available.
- [x] 6.3 Review the final diff for accidental runtime files, generated artefacts, secrets, and unrelated changes.

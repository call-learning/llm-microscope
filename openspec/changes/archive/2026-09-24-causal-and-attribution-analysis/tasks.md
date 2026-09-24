## 1. Attribution Methods

- [x] 1.1 Define a common signed attribution result structure containing token position, token label, raw score, magnitude, selected output token, and method; verify existing Gradient x Input output can be represented without changing its current behavior.
- [x] 1.2 Add optional Integrated Gradients using the Captum extra with configurable steps and progress feedback; verify the core attribution view still works when Captum is absent.
- [x] 1.3 Add method selection and comparison controls for the same prompt and target token; verify each displayed chart and table identifies its attribution method.
- [x] 1.4 Replace absolute-only attribution rendering with signed diverging bars and signed table values; verify positive, negative, and zero contributions remain distinguishable.
- [x] 1.5 Add attribution limitations and optional-dependency guidance to `microscope/docs.py`; verify missing Captum produces an actionable message rather than an empty result.

## 2. Probe Diagnostics

- [x] 2.1 Extend probe evaluation data to retain class counts, training scores, held-out scores, baseline scores, and evaluation metadata; verify existing probe fitting remains usable.
- [x] 2.2 Add deterministic stratified held-out evaluation when class sizes permit and explicit training-only fallback for small datasets; verify both paths with balanced and undersized fixtures.
- [x] 2.3 Add baseline reference, class-count display, imbalance warnings, and train-versus-held-out layer charts; verify a majority-class baseline is visible.
- [x] 2.4 Add repeated-split variation or confidence intervals only when sample size supports it; verify uncertainty is labeled and omitted with an explanatory warning when unsupported.

## 3. Patching Matrix

- [x] 3.1 Add metadata storage for model, source prompt, target prompt, patch target, attention head, metric, and run settings; verify metadata survives Streamlit reruns for the current session.
- [x] 3.2 Add compatibility filtering and matrix assembly for patch layer by patch target or metric; verify incompatible runs are not combined and missing cells remain missing.
- [x] 3.3 Render the patching matrix beside the existing curve and detail table with baseline and experiment context; verify single-target and multi-target results are both understandable.
- [x] 3.4 Add causal interpretation guidance and tests for matrix values against score-curve deltas; verify the displayed metric matches the selected logit or probability delta.

## 4. Validation

- [x] 4.1 Add focused tests for signed attribution, optional Captum behavior, probe splits and warnings, and patching compatibility/matrix assembly; verify the project test command passes.
- [x] 4.2 Run Python compilation, documentation/import checks, and strict OpenSpec validation for `causal-and-attribution-analysis`; verify no syntax or specification errors remain.
- [x] 4.3 Run a Streamlit smoke test covering attribution, probe, and patching paths with the default model; verify failures in optional methods do not prevent other tabs from loading.

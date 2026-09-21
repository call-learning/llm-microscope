## Why

The workbench currently has strong *causal* tools (patching, attribution) but no first-class *observational* tool for the classic question: "at which layer does a property of the prompt become linearly decodable from the hidden state?" A linear probe fills that gap and pairs with the patching score curve — the probe can say "decodable at layer L" and the curve can confirm "intervening at layer L changes the output."

## What Changes

- Add a **Linear Probe** tab that fits a binary linear classifier on the final-token hidden state at each decoder layer and reports per-layer accuracy, showing where the property becomes decodable.
- Provide **two label modes**, both feeding one shared fitting core:
  - **Concept mode**: user supplies a positive and a negative prompt template plus a comma-separated list of fill values (e.g. `The capital of {X} is` vs `The largest city in {X} is`, filled with `France, Germany, …`), producing a balanced contrastive training set that decodes the *concept*, not the surface string.
  - **Quick (surface) mode**: user supplies a property substring; each auto-generated prompt is labeled by whether it contains the substring. Decodes the *surface form*.
- Use **L2-regularized logistic regression** (scikit-learn, already a dependency) with an exposed penalty-strength slider.
- Report the **learned weight vector** (the probe direction) alongside accuracy, so a single probe fit yields both the accuracy curve and a reusable direction.
- Explicitly out of scope for this change: a steering / activation-addition UI (add scaled direction to an activation and observe the effect) is deferred to a follow-on change.

## Capabilities

### New Capabilities
- `linear-probing`: observational linear-probe view — fit an L2 logistic-regression binary classifier on per-layer final-token hidden states using contrastive (concept) or substring (quick) labels, and report per-layer accuracy plus the learned direction vector.

### Modified Capabilities
<!-- None. No existing capability is being modified; the project has no baseline specs yet. -->

## Impact

- New `microscope/probing.py` — label generation (concept + quick), training-set assembly, per-layer probe fitting, and result struct.
- `app.py` — new Linear Probe tab (mode selector, template/substring inputs, fill-value input, penalty slider, accuracy curve, direction display).
- Reuses existing `Runtime` and `analyse()` (hidden states are already captured) — no change to the runtime or analysis data flow.
- No new runtime dependencies (scikit-learn is already a dependency).
- Memory: the probe fits on CPU float32 hidden states already held by `AnalysisResult`; no additional GPU pressure.

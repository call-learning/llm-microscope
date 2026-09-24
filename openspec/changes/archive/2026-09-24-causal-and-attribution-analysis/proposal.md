## Why

LLM Microscope can already run basic activation patching, Gradient x Input attribution, and linear probes, but each result is presented in isolation and important reliability context is missing. Users need comparable intervention views, alternative attribution methods, and probe diagnostics to distinguish a useful signal from an unstable or misleading explanation.

## What Changes

- Add a patching effect matrix that compares patch layers and supported patch targets while retaining the current score curve.
- Add Integrated Gradients and/or occlusion attribution as explicit alternative methods to Gradient x Input.
- Separate positive and negative attribution contributions and allow method comparison for the same prompt and target token.
- Add linear-probe train/test accuracy, chance and baseline references, confidence information, and warnings for small or imbalanced datasets.
- Add explanatory copy describing the limitations of causal interventions, gradient attributions, occlusion, and probes.
- Keep experiments on demand and preserve the existing model and architecture compatibility behavior.

## Capabilities

### New Capabilities

- `attribution-analysis`: Alternative attribution methods and comparable positive/negative token contribution views.
- `probe-diagnostics`: Reliability and evaluation diagnostics for layer-wise linear probes.

### Modified Capabilities

- `activation-patching`: Add comparative patching matrices and preserve the existing causal score curve and supported-target behavior.

## Impact

- Affected `app.py`, `microscope/interventions.py`, `microscope/probing.py`, and attribution-related analysis code.
- Captum may become an optional dependency for Integrated Gradients; the core application must continue to work without it.
- Additional forwards and gradient/occlusion passes can be expensive, so the UI must remain explicitly on demand and show progress.
- Existing patching results and probe workflows remain available; this change adds diagnostics rather than replacing them.

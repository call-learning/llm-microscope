# Design

## Context

The Toolbox currently discovers optional packages with `find_spec`, while native analysis owns model execution and Plotly rendering. Tuned Lens, BertViz, and CircuitsVis are direct dependencies; TransformerLens, NNsight, Captum, and UMAP are optional extras; SAELens and Pyvene are not currently installed. The existing specifications require optional failures to be isolated.

## Decisions

### 1. One adapter boundary

Each provider gets a lazy adapter with `probe`, `prepare`, and `render` responsibilities. Adapters return metadata and view-ready results; they do not run during application startup.

### 2. Native data remains canonical

Adapters consume the existing `AnalysisResult`, component results, or explicit experiment inputs. They do not silently convert the model into another wrapper. Wrapper-based tools receive a separate explicit model-load path and are never used as an automatic fallback.

### 3. Browser output is optional

BertViz and CircuitsVis rendering is attempted only after a provider compatibility check. If Streamlit embedding is not supported, the adapter reports that state and the native Plotly view remains authoritative.

### 4. Extras stay isolated

SAELens and Pyvene are added as named optional extras, not base dependencies. All provider imports remain lazy and their version/API boundaries are reported.

### 5. No fake compatibility

Successful import is not sufficient for adapter readiness. Model family, tensor shape, artefact, frontend, and device requirements are checked independently.

## Risks

- Third-party APIs may change -> constrain versions and use focused smoke probes.
- Wrapper conversion can duplicate model memory -> report memory expectations and run on demand.
- Frontend components can fail in Streamlit -> retain native Plotly views and actionable messages.

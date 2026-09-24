# Tasks

## 1. Optional tool status model

- [x] 1.1 Extend the optional-tool metadata and status rows to distinguish package installation, compatibility status, and application-adapter availability for SAELens, Pyvene, BertViz, and CircuitsVis; verify with focused unit tests covering installed, missing, and non-integrated states.
- [x] 1.2 Add capability and limitation metadata for each candidate visualisation tool, including its output category and frontend, model, or memory boundaries; verify every candidate produces a complete Toolbox status row.

## 2. Isolated compatibility reporting

- [x] 2.1 Add explicit, isolated status/compatibility handling for candidate tools without importing optional packages during normal application startup; verify the core application and native views remain importable when candidate packages are absent.
- [x] 2.2 Ensure unsupported models, missing model-specific artefacts, and provider errors return actionable unavailable-state messages rather than empty or fabricated visualisations; verify with tests for each failure category.

## 3. Toolbox presentation

- [x] 3.1 Update the Toolbox interface to display installed, compatible, and integrated states separately and to explain when a candidate has no application adapter; verify the rendered status table distinguishes all states.
- [x] 3.2 Document the assumptions and limitations of any optional visualisation output shown by the Toolbox; verify the displayed metadata identifies attention, circuit, activation, or sparse-feature scope where applicable.

## 4. Core visualisation preservation

- [x] 4.1 Confirm optional-provider failures do not invalidate the current analysis or disable native Plotly views; verify with an integration test that a simulated provider failure leaves other tabs and views available.
- [x] 4.2 Keep candidate dependencies out of the base dependency set until an adapter is accepted; verify `pyproject.toml` and the lockfile contain no new candidate dependency and run the existing environment checks.

# Design

## Context

The core application currently uses Hugging Face model outputs and Plotly for its built-in views. `microscope/toolbox.py` already maintains a registry of optional tools, detects packages lazily, and reports capability and limitation metadata. SAELens, Pyvene, BertViz, and CircuitsVis are listed as evaluation candidates, but none currently has an application adapter or an optional dependency entry.

The existing `advanced-tool-integrations` and `analysis-visualizations` specifications require optional integrations to remain isolated from the core workbench. This design extends that boundary without changing the native analysis data flow.

## Goals / Non-Goals

**Goals:**

- Represent installation, compatibility, and application-adapter status separately.
- Keep candidate integrations discoverable without implying that package installation provides an application feature.
- Establish a common boundary for future attention, circuit, activation, and sparse-feature adapters.
- Preserve current Plotly visualisations and graceful error handling when an optional provider is absent or fails.

**Non-Goals:**

- Add any of the four candidate packages in this change.
- Implement adapters or embed third-party frontend components.
- Convert the current Hugging Face model to another library's model wrapper automatically.
- Replace existing native attention, activation, attribution, or patching views.

## Decisions

### 1. Keep the candidate registry as the discovery boundary

The existing Toolbox registry remains the single discovery surface for optional tools. Candidate metadata should describe the output type, package name, adapter state, and limitations. This avoids eager imports and keeps the main app independent of optional packages.

**Alternative considered:** Import each package during application startup. Rejected because one incompatible optional package could prevent the core Streamlit app from starting.

### 2. Separate package presence from adapter availability

Tool status should expose at least three independent facts: whether the package can be imported, whether the selected model is compatible, and whether this application has an adapter capable of producing a view. Compatibility checks remain explicit and isolated; an installed package without an adapter remains an evaluation candidate.

**Alternative considered:** Treat successful import as support. Rejected because model wrappers, tensor formats, frontend rendering, and architecture support are separate compatibility boundaries.

### 3. Use adapter-owned visualisation contracts

When a candidate becomes integrated, its adapter should return a view-ready result together with labels for the represented tokens, layers, heads, or features and a concise description of assumptions. Rendering failures should be caught at the optional-provider boundary and reported as an unavailable view rather than propagated into unrelated tabs.

**Alternative considered:** Let each third-party library render directly into arbitrary application pages. Rejected because it makes error handling, accessibility, data identification, and frontend dependencies inconsistent.

### 4. Do not add optional dependencies until an adapter is accepted

Dependency metadata and lockfile changes are deferred until a specific integration is selected for implementation. The specification applies equally to all four candidates, while future implementation changes can choose one tool and validate its version and model support independently.

## Risks / Trade-offs

- **Third-party APIs and model wrappers evolve quickly** -> Pin or constrain each optional dependency only when its adapter is implemented, and keep compatibility checks explicit.
- **Interactive third-party visualisations may add frontend or browser requirements** -> Report those requirements in Toolbox metadata and preserve a native fallback where available.
- **Sparse-feature and circuit tools may require model-specific artefacts** -> Treat missing artefacts and unsupported architectures as compatibility failures, not empty results.
- **Multiple status dimensions may be more complex for users** -> Present clear labels such as installed, compatible, integrated, or unavailable rather than collapsing them into one boolean.

## Migration Plan

No migration is required. Existing candidate entries remain non-integrated until an adapter and dependency are deliberately added. Existing native views and optional integrations continue to operate unchanged.

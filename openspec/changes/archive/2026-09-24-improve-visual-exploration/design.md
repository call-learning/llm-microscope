## Context

The analysis result already retains CPU copies of final logits, hidden states for every layer, token labels, and optional attention maps. The current UI mostly exposes those values as tables or one-dimensional Plotly charts. The application is memory-constrained and runs synchronously in Streamlit, so visualisations should derive from the retained result whenever possible and perform additional work only when the user requests it.

The existing patching score curve returns one row per layer and one selected patch target per run. The new matrix view therefore needs to compare result sets without changing the established patching experiment or its forward-pass budget.

## Goals / Non-Goals

**Goals:**

- Add a focused prediction-exploration view that relates token positions, layers, candidate predictions, uncertainty, and representation size.
- Make candidate-token selection explicit and let users follow a candidate through layers.
- Add representation heatmap or trajectory views using data already present in the analysis result.
- Add a patching heatmap that can compare completed score-curve results and clearly labels its metric and baseline.
- Keep explanations and interpretation warnings close to each chart.
- Keep expensive work on demand, avoid unnecessary model forwards, and preserve current model compatibility.

**Non-Goals:**

- This change will not claim that attention, attribution, or logit-lens views are complete explanations.
- It will not add sparse autoencoders, automated circuit discovery, neuron labeling, or a new external visualisation framework.
- It will not change the model-loading contract or replace the existing analysis tabs.
- It will not silently run a full patching sweep for every patch target; comparisons require explicit user requests.

## Decisions

### 1. Add a dedicated prediction exploration tab

Add a new tab, tentatively named `Prediction journey`, rather than putting every chart into `Tokens & prediction`. The existing tab remains a simple entry point for tokenisation and final next-token probabilities; the new tab can contain controls for selected candidate token, metric, layer, and token position without making the basic view overwhelming.

Alternative considered: add all charts to the existing tab. This minimizes navigation but mixes beginner-oriented output with expensive and technical controls, making the tab harder to teach and maintain.

### 2. Derive layer predictions from hidden states

For each selected layer, apply the runtime's final normalization and language-model head to the hidden state at each token position. Build a long-form table containing layer, position, token label, candidate token ID, probability, logit, rank, and entropy. Compute this lazily for the selected candidate and view rather than storing a full vocabulary-sized tensor for every layer.

The final-token logit-lens implementation can be reused conceptually, but the new helper should support all positions and selected-token metrics. The final-token top-k list remains the source for the candidate selector, with an optional token-ID input for candidates not in the top-k list.

Alternative considered: run a new model forward for every layer. This would be much slower and would duplicate information already present in hidden states.

### 3. Use Plotly heatmaps and linked controls

Use Plotly for heatmaps, bars, and lines, matching the existing application. The primary prediction view will include:

- a token-position by layer heatmap for the selected token's probability, rank, or logit;
- a line chart showing the selected token across layers at a chosen position;
- an entropy line or heatmap for uncertainty;
- a final-position top-k bar chart.

Token labels and layer numbers must be included in axis labels or hover data. Long prompts should use position labels in the axis and full token text in hover data to avoid unreadable charts.

Alternative considered: a graph or Sankey diagram. These imply discrete flow between states and could suggest a stronger causal story than the measurements support.

### 4. Add representation views to the Activations tab

Keep the existing norm chart and PCA/UMAP controls. Add a token-by-layer heatmap based on hidden-state norms or layer-to-layer cosine change, plus a selectable-token trajectory across layers. Use clear metric selectors and explain that PCA/UMAP geometry is approximate.

Compute scalar metrics from CPU hidden states. Do not add a persistent all-pairs distance tensor; calculate the selected layer's pairwise cosine matrix only when requested.

Alternative considered: replace PCA/UMAP with one new projection method. Keeping both preserves current exploratory behavior and avoids treating any single projection as authoritative.

### 5. Keep patching experiments separate from patching presentation

Store the latest score-curve result in session state as today, and add an explicit comparison control for results from multiple patch targets. Convert selected result frames into a matrix with patch layer on one axis and target or metric on the other, then render a heatmap. If only one target has been run, show a one-row heatmap and explain that cross-target comparison requires additional runs.

The matrix must use the existing stable baseline-top-token deltas (`delta_logit` or `delta_prob`) and retain the current curve and detail table. Unsupported targets are omitted from selectable comparisons and described as unavailable, never filled with zeroes.

Alternative considered: change `patching_score_curve` to run all targets in one call. That would increase memory and runtime unexpectedly and would make the current single-target workflow less predictable.

### 6. Centralize explanatory copy in `microscope/docs.py`

Add descriptions, metric definitions, and interpretation warnings to the existing docs source of truth. `app.py` should only select and render documentation, keeping the wording editable independently from layout and computation.

### 7. Reuse analysis identity and on-demand state

Associate derived visual data with the current analysis prompt and model configuration in session state. Clear or ignore derived data when the prompt or model changes. Views derived from the current `AnalysisResult` should render without a new spinner; additional inference or expensive transformations should run behind a progress indicator and be cached for the current inputs.

## Risks / Trade-offs

- [Full-vocabulary projection is expensive] -> Project only the selected layer/position data needed by the current view, compute lazily, and provide a clear warning for long prompts or large models.
- [Heatmaps become unreadable for long prompts] -> Use token positions on axes, full token text in hover data, and retain the existing prompt-length limit.
- [A logit-lens probability can be misread as a literal intermediate prediction] -> Put the existing logit-lens caveat beside the new charts and label the operation as a diagnostic decode.
- [Patching comparisons may mix incompatible experiments] -> Store source prompt, target prompt, model name, patch target, and metric with each result; only compare compatible prompt pairs and model settings.
- [Streamlit reruns can lose expensive results] -> Keep derived frames in session state keyed by the analysis identity and explicit view controls.
- [Projection artifacts may look like semantic clusters] -> Keep metric definitions and PCA/UMAP limitations visible next to the chart.

## Migration Plan

No data migration is required. Implement the new derived-data helpers and views behind the existing analysis flow, then add documentation and tests. Existing session-state keys and tabs remain valid. If a new view causes runtime or memory problems, hide it while retaining the existing views and score curve; no persisted user data needs rollback.

## Open Questions

- Whether the new prediction view should initially expose probability and rank only, or also raw logits by default. The metric selector can support all three while choosing probability as the initial default.
- Whether patching result comparison should persist multiple runs in the current browser session or begin with an explicit upload/export path. The first implementation can use session state only.

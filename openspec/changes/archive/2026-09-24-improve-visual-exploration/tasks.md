## 1. Analysis Data Helpers

- [x] 1.1 Add lazy helpers for layer-by-layer candidate probability, logit, rank, and entropy across token positions, reusing retained hidden states; verify results against final-layer logits on a short prompt.
- [x] 1.2 Add helpers for token-by-layer representation metrics, including hidden-state norm and layer-to-layer cosine change; verify output dimensions and token labels for prompts of different lengths.
- [x] 1.3 Add selected-token and analysis-identity state handling so derived results are invalidated when the prompt or model changes; verify no stale chart is shown after a new analysis.

## 2. Prediction Journey UI

- [x] 2.1 Add a dedicated prediction-exploration tab with controls for candidate token, metric, and token position; verify it appears after the existing prediction tab without changing current tab behavior.
- [x] 2.2 Add the token-by-layer heatmap and selected-position layer chart using Plotly, with readable layer/position axes and token hover labels; verify the charts render for the default prompt.
- [x] 2.3 Add the final next-token probability bar chart and entropy view, defaulting to probability while allowing rank/logit selection; verify candidate order and displayed values match the analysis data.
- [x] 2.4 Add empty, unavailable, and long-prompt handling for prediction views with explanatory messages; verify a zero-result or insufficient projection does not crash the page.

## 3. Activation Visualisations

- [x] 3.1 Add a token-by-layer activation heatmap with a metric selector and token-position context to the Activations tab; verify it uses the current analysis result without additional model inference.
- [x] 3.2 Add a selectable-token layer trajectory and optional pairwise cosine-distance view; verify the selected token and layer are reflected in labels and hover data.
- [x] 3.3 Add metric definitions and PCA/UMAP interpretation guidance beside the new activation charts; verify the guidance is visible without opening source code.

## 4. Patching Comparison View

- [x] 4.1 Extend patching session state to retain compatible score-curve results with source prompt, target prompt, model, patch target, and metric metadata; verify incompatible runs are not combined.
- [x] 4.2 Add an explicit patch-target comparison control and convert compatible score-curve results into a layer-by-target effect matrix; verify missing or unsupported targets are not rendered as zero values.
- [x] 4.3 Render the patching effect heatmap alongside the existing curve and detail table, with logit/probability metric labels and baseline/source/target context; verify a single-target result remains useful.
- [x] 4.4 Add causal-interpretation guidance for the curve and heatmap; verify it explains baseline comparison and distinguishes causal influence from concept attribution.

## 5. Documentation and Validation

- [x] 5.1 Add all new tab, chart, metric, and interpretation copy to `microscope/docs.py`; verify `app.py` contains no duplicated explanatory copy for these views.
- [x] 5.2 Add focused tests for probability/rank/entropy calculations, representation metric shapes, and patching matrix assembly; verify the project test command passes.
- [x] 5.3 Run Python compilation, documentation/import checks, and `openspec validate --change improve-visual-exploration`; verify no syntax, whitespace, or specification errors remain.
- [x] 5.4 Run a Streamlit smoke test with the default model and prompt, inspect every new chart at desktop and narrow widths, and verify existing tabs still load.

## Why

LLM Microscope currently exposes useful measurements, but several are presented as isolated tables or single curves. Users can see individual predictions and layer values without an integrated view of how token representations, uncertainty, and candidate predictions develop through the model. A focused visual exploration layer will make the existing analyses easier to interpret and compare without requiring users to export data or already know which metric to inspect.

## What Changes

- Add a token-by-layer exploration dashboard that combines token positions, candidate-token probability, candidate rank, entropy, and hidden-state measures.
- Add selectable-token analysis so users can follow any candidate token across layers instead of only the final top prediction.
- Add clearer uncertainty visualisations, including next-token probability bars, entropy by position/layer, and rank evolution.
- Add activation visualisations that complement PCA/UMAP with token-by-layer heatmaps, cosine-distance views, and layer trajectories.
- Add a patching heatmap or equivalent matrix view for comparing intervention effects across source layers and patch targets, while retaining the existing score curve.
- Add interpretation guidance and metric definitions directly alongside the new visualisations.
- Preserve on-demand execution, current model support, and the existing attention, attribution, probing, and patching workflows.

## Capabilities

### New Capabilities

- `analysis-visualizations`: Integrated visual exploration of token predictions, uncertainty, hidden-state evolution, and representation similarity across model layers.

### Modified Capabilities

- `activation-patching`: Extend patching results with a comparative matrix/heatmap representation while retaining the existing per-layer score curve and on-demand execution behavior.

## Impact

- Affected Streamlit tab layout and explanatory documentation in `app.py` and `microscope/docs.py`.
- Affected analysis data preparation in `microscope/analysis.py`, including metrics for entropy, token rank, selected-token evolution, and representation comparisons.
- Affected patching presentation and possibly score-aggregation helpers in `microscope/interventions.py`.
- Plotly remains the primary visualisation dependency; no additional dependency is required for the initial dashboard unless a later design explicitly selects an interactive graph library.
- Existing model inference and tensor-memory constraints remain important: expensive views must stay on-demand and should reuse cached analysis results where possible.

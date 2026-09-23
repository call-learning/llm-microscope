"""Editable documentation for every LLM Microscope view.

This module is the single source of truth for the explanatory copy shown in
each tab: the "About this view" text (what / why) and the "Suggested actions"
that appear in the popover. Edit the strings here to change the wording
without touching the page code in ``app.py``.

Each entry is keyed by the tab it documents. ``examples`` is a list of dicts:
    - ``label``: button text shown in the popover.
    - ``set``: mapping of Streamlit session-state keys to values. Clicking the
      button writes these into ``st.session_state`` via an ``on_click``
      callback, so the page's widgets are pre-filled on the next rerun.
    - ``run``: optional trigger name. When present, the callback also sets
      ``st.session_state["trigger_<run>"]`` so the page can fire the relevant
      analysis automatically. Leave it out for examples that only pre-fill
      the widgets.

The ``set`` keys must match the ``key=`` arguments of the corresponding
widgets in ``app.py``. Because the write happens in an ``on_click`` callback,
the value is applied before the widgets are (re)instantiated — this is the
only way Streamlit allows setting a keyed widget's value at runtime.
"""

from __future__ import annotations

# Top-of-page copy (not tied to a single tab).
APP_TITLE = "🔬 LLM Microscope"
APP_SUBTITLE = "Local interpretability workbench — logit lens, attention, activations, attribution and causal patching"
SIDEBAR_MODEL_CAPTION = "Changing the model reloads it. Start with 1.7B on a 12 GB GPU."

# Default prompt shown when the user has not typed anything. An empty prompt
# tokenizes to zero tokens, which crashes the model's attention reshape, so the
# text area falls back to this value instead of submitting an empty string.
DEFAULT_PROMPT = "The capital of France is"

# Keyed by tab. Order / names mirror the tab labels in app.py.
VIEWS: dict[str, dict] = {
    "Tokens & prediction": {
        "what": "Shows how the model splits your prompt into tokens (with their IDs) and the top-20 tokens it expects to come next from the final layer.",
        "why": "Token IDs are the raw unit the model actually reads, while the next-token distribution is its direct prediction — the ground truth every other view interprets.",
        "examples": [
            {
                "label": "Predict the next word of a simple fact",
                "set": {"prompt": "The capital of France is"},
                "run": "analyse",
            },
            {
                "label": "Watch code completion",
                "set": {"prompt": "def fibonacci(n):\n    if n <= 1:\n        return "},
                "run": "analyse",
            },
        ],
    },
    "Logit lens": {
        "what": "Decodes the model's prediction after every layer by applying the final normalization + output head to each intermediate residual state. The table has one row per layer; the columns `#1 token` … `#5 token` are the five most likely next tokens *as decoded at that layer*, each paired with its `#N probability` (the softmax probability of that token). `#1` is the layer's top prediction, `#2` the runner-up, and so on.",
        "why": "Reveals *when* the model commits to its answer: early layers tend to be noisy, while the answer usually emerges and stabilises in the middle-to-late layers. Watch the `#1 token` column down the rows to see the answer appear, and use `#2`–`#5` to spot competing candidates the model is weighing before it settles. The line chart below tracks the probability of the final (last-layer) top token across layers.",
        "examples": [
            {
                "label": "Track a fact answer appearing",
                "set": {"prompt": "The capital of France is"},
                "run": "analyse",
            },
            {
                "label": "See an ambiguous choice resolve",
                "set": {"prompt": "Paris is known for its"},
                "run": "analyse",
            },
        ],
    },
    "Attention": {
        "what": "Per-head attention heatmaps for the selected layer: how strongly each query token attends to every key token.",
        "why": "Attention is a quick (but incomplete) look at which tokens the model is comparing. Diagonals copy information; off-diagonals reveal syntactic or factual links. Use it to spot where a token 'looks' at the token that should inform it.",
        "examples": [
            {
                "label": "Subject → verb agreement",
                "set": {"prompt": "The keys that hang on the hook are"},
                "run": "analyse",
            },
            {
                "label": "Where does the answer come from?",
                "set": {"prompt": "Qwen was developed by Alibaba. Who made Qwen?"},
                "run": "analyse",
            },
        ],
    },
    "Activations": {
        "what": "Two views of the hidden (residual) states: the per-layer vector norm, and a 2-D projection (PCA or UMAP) of the tokens' hidden state at a chosen layer.",
        "why": "Norms show how much 'signal' is still being built up as layers progress. The projection clusters tokens with similar representations, so you can see whether, say, all country names land in one region of the space.",
        "examples": [
            {
                "label": "Cluster a list of countries",
                "set": {"prompt": "France Germany Japan Spain Italy Brazil"},
                "run": "analyse",
            },
        ],
    },
    "Compare A/B": {
        "what": "Runs two prompts and plots the cosine similarity of their final-token hidden states at each layer.",
        "why": "High similarity at a layer means the model represents the two prompts in much the same way there; the point where the curves diverge tells you where the model starts treating them differently.",
        "examples": [
            {
                "label": "Two different capitals",
                "set": {
                    "prompt": "The capital of France is",
                    "prompt_b": "The capital of Germany is",
                },
                "run": "compare",
            },
            {
                "label": "Same fact, different phrasing",
                "set": {
                    "prompt": "What is 2 + 2?",
                    "prompt_b": "What is the sum of two and two?",
                },
                "run": "compare",
            },
        ],
    },
    "Patching": {
        "what": "Causal intervention: copies or ablates the final-token state at a layer (or a specific module: residual stream, attention output, or MLP output) and compares the resulting prediction to the baseline.",
        "why": "Unlike correlational views, patching shows *causal* influence — if swapping a state changes the output, that state was doing real work. The score curve locates *which* layer matters most.",
        "examples": [
            {
                "label": "Score curve, residual stream",
                "set": {
                    "patch_source": "The capital of France is",
                    "patch_target": "The capital of Germany is",
                    "mode": "Score curve (all layers)",
                    "patch_target_kind": "Residual",
                },
                "run": "patch_curve",
            },
            {
                "label": "Ablate the middle layer (zero)",
                "set": {
                    "patch_source": "The capital of France is",
                    "patch_target": "The capital of Germany is",
                    "mode": "Single layer",
                    "ablation": "Zero",
                    "patch_target_kind": "Residual",
                },
                "run": "patch_single",
            },
        ],
    },
    "Attribution": {
        "what": "Gradient × input: multiplies each input token's embedding by the gradient of the selected output logit, scoring how much each token contributed to that prediction.",
        "why": "A fast, local explanation of *which* tokens drove a specific answer. Positive scores push the token up; negative scores push it down. It is sensitive to the chosen output token.",
        "examples": [
            {
                "label": "What made it say 'Paris'?",
                "set": {"prompt": "The capital of France is", "attr_target": ""},
                "run": "attribution",
            },
        ],
    },
    "Linear Probe": {
        "what": "Fits a binary logistic regression on the final-token hidden state at each layer, then reports per-layer accuracy and the probe's direction vector.",
        "why": "If a simple linear probe can read a property (e.g. 'the country is France') from a layer's representation, that property is linearly decodable there — a cheap way to locate where knowledge or a feature becomes available.",
        "examples": [
            {
                "label": "Capital vs. largest city (concept)",
                "set": {
                    "probe_mode": "Concept (contrastive)",
                    "pos_template": "The capital of {X} is",
                    "neg_template": "The largest city in {X} is",
                    "fill_values": "France, Germany, Japan, Spain, Italy, Brazil",
                },
                "run": "probe_concept",
            },
            {
                "label": "Quick substring probe",
                "set": {
                    "probe_mode": "Quick (substring)",
                    "substring": "France",
                    "quick_prompts": "The capital of France is\nThe capital of Germany is\nThe capital of Japan is\nThe largest city in France is\nThe largest city in Germany is\nThe largest city in Japan is",
                },
                "run": "probe_quick",
            },
        ],
    },
    "Toolbox": {
        "what": "Status of the optional research integrations (Captum, UMAP, NNsight, TransformerLens) and a CPU-only compatibility probe for TransformerLens.",
        "why": "These libraries add deeper tools but evolve fast and can constrain model/PyTorch versions. The main analysis pages intentionally do not depend on NNsight or TransformerLens, so a missing install never blocks the core views.",
        "examples": [],
    },
}


def view(name: str) -> dict:
    """Return the documentation entry for a tab, with empty defaults."""
    entry = VIEWS.get(name)
    if entry is None:
        return {"what": "", "why": "", "examples": []}
    return {
        "what": entry.get("what", ""),
        "why": entry.get("why", ""),
        "examples": list(entry.get("examples", [])),
    }

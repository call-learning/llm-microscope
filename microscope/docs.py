"""Editable documentation for every LLM Microscope view.

This module is the single source of truth for the explanatory copy shown in
the app and each tab: the app guide, "About this view" text (what / why /
prior knowledge), and the "Suggested actions"
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
APP_GUIDE = """
**What is this?** LLM Microscope is a learning and investigation workbench for
decoder-only large language models (LLMs). It runs a prompt through a model and
exposes some of the intermediate numbers produced inside it: tokens, hidden
states, attention, layer outputs, gradients, and predictions.

**What can it help you understand?** A language model does not look up one
answer in a database. It repeatedly transforms a sequence of token vectors
through layers, then converts the final vector into probabilities for the next
token. The tabs provide different perspectives on that process: what the model
read, what it predicts, when a prediction appears, which representations are
similar, and whether changing an internal state changes the result.

**How much prior knowledge is needed?** No machine-learning background is
needed for **Tokens & prediction**. It helps to know that a *token* is a piece
of text and that a *probability* is the model's relative preference for a
possible next token. For **Logit lens**, **Attention**, **Activations**, and
**Compare A/B**, a basic idea of neural-network layers and vectors is useful.
For **Patching**, **Attribution**, and **Linear Probe**, some familiarity with
causal experiments, gradients, or classification will make the results easier
to interpret. The help panel in every tab explains the terms used there.

**A good first path:** run the default prompt, read **Tokens & prediction**,
then open **Logit lens** to see how the answer develops across layers. Use
**Attention** to inspect token-to-token relationships, **Compare A/B** to
compare prompts, and only then try **Patching**, **Attribution**, or
**Linear Probe**. These are evidence about the model's computation, not a
complete human-readable explanation. In particular, attention is not proof of
causation, and probabilities are not claims of certainty.
"""

# Default prompt shown when the user has not typed anything. An empty prompt
# tokenizes to zero tokens, which crashes the model's attention reshape, so the
# text area falls back to this value instead of submitting an empty string.
DEFAULT_PROMPT = "The capital of France is"

# Keyed by tab. Order / names mirror the tab labels in app.py.
VIEWS: dict[str, dict] = {
    "Tokens & prediction": {
        "what": "Shows the pieces of text the tokenizer gives the model and the model's top 20 possible next tokens, with their IDs and probabilities.",
        "why": "This is the baseline for every other tab: it shows exactly what was read and what the model predicts before we inspect how that prediction was formed.",
        "know": "None. A token is simply a piece of text, which may be a whole word, part of a word, punctuation, or whitespace. A probability here is a preference among possible next tokens, not a guarantee.",
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
    "Prediction journey": {
        "what": "Follows one candidate next token through every layer and prompt position. It combines probability, rank, logit, and entropy with heatmaps and layer-by-layer charts.",
        "why": "Shows when a candidate becomes plausible, whether it wins against competitors, and where the model is more or less uncertain while processing the prompt.",
        "know": "A logit is an unnormalised score, rank counts how many candidates score higher, and entropy measures how spread out the full next-token distribution is. These layer-wise values are diagnostic decodes, not literal intermediate predictions.",
        "examples": [
            {
                "label": "Follow the capital answer through layers",
                "set": {"prompt": "The capital of France is"},
                "run": "analyse",
            },
        ],
    },
    "Logit lens": {
        "what": "Decodes the model's prediction after every layer by applying the final normalization + output head to each intermediate residual state. The table has one row per layer; the columns `#1 token` … `#5 token` are the five most likely next tokens *as decoded at that layer*, each paired with its `#N probability` (the softmax probability of that token). `#1` is the layer's top prediction, `#2` the runner-up, and so on.",
        "why": "Reveals when a candidate answer becomes visible in the computation. Watch `#1 token` down the rows, and use the other ranked tokens to see competing candidates. The chart follows the final top token's probability across layers.",
        "know": "It helps to know that a layer is one processing stage and that a hidden state is a vector. This is a diagnostic projection: the model does not literally make a finished prediction at every layer.",
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
        "what": "Per-head attention heatmaps, entropy summaries, raw-pattern head comparisons, and an optional rollout aggregate for the selected layer and heads.",
        "why": "Shows which earlier tokens each token is weighting while it is processed and whether a head is concentrated or diffuse. It can suggest relationships such as a verb attending to its subject, but it does not by itself prove that one token caused the answer.",
        "know": "Each row is a query token and each column is a key token. Entropy describes concentration over key positions. Rollout averages heads, adds a residual identity, and composes layers; it is a descriptive summary, not a causal explanation.",
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
        "what": "Views the hidden (residual) states as layer-by-token heatmaps, token trajectories, pairwise cosine distances, and 2-D PCA or UMAP projections.",
        "why": "These views show how token representations change and which tokens are similar at a layer. The heatmap and trajectory make layer changes visible before using a dimensionality-reduction projection.",
        "know": "It helps to know that distances in a 2-D projection are approximate: PCA preserves broad variance and UMAP emphasises local neighbourhoods. A cluster is a pattern to investigate, not automatically a discovered concept.",
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
        "why": "Shows where two prompts have similar or different final-token representations. A divergence can indicate that their processing is separating, while similarity does not mean the prompts have identical meanings.",
        "know": "You need the basic idea of a vector and cosine similarity: 1 means the vectors point in the same direction, 0 means no directional similarity, and -1 means opposite directions.",
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
        "what": "Causal intervention: copies or ablates the final-token state at a layer (or a specific module: residual stream, attention output, or MLP output) and compares the resulting prediction to the baseline. Compatible score curves can also be compared in a patch-target heatmap.",
        "why": "Unlike a correlation, an intervention tests whether changing an internal state changes the output. The score curve and heatmap help locate layers and targets where the source prompt affects the target prompt.",
        "know": "Some causal-experiment vocabulary is useful. The source supplies a state, the target supplies the context, and the baseline is the unmodified target. A changed prediction is evidence for influence, not automatically a human-readable explanation.",
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
        "what": "Compares token-level Gradient x Input and optional Integrated Gradients scores for a selected output token, preserving positive and negative contributions.",
        "why": "Provides local evidence about which input tokens support or oppose a specific output token under each attribution method. Comparing methods can reveal instability, but agreement is not proof of a complete causal explanation.",
        "know": "You need only the intuition that a gradient measures sensitivity. Integrated Gradients uses a baseline and many interpolation steps, so it is slower and may require the optional Captum package.",
        "examples": [
            {
                "label": "What made it say 'Paris'?",
                "set": {"prompt": "The capital of France is", "attr_target": ""},
                "run": "attribution",
            },
        ],
    },
    "Linear Probe": {
        "what": "Fits a binary logistic regression on the final-token hidden state at each layer and reports training, held-out, baseline, and uncertainty information alongside the probe direction.",
        "why": "Tests whether a simple classifier can read a chosen distinction from each layer's representation while showing whether the score generalizes beyond the training examples.",
        "know": "Basic classification vocabulary helps. Positive and negative examples must be balanced and meaningful; otherwise the probe may learn a shortcut in the data rather than the intended concept. Small datasets may only support training-only results.",
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
        "what": "Status, capabilities, limitations, and explicit compatibility checks for optional research integrations such as Captum, UMAP, NNsight, and TransformerLens, plus evaluation candidates.",
        "why": "Shows which optional libraries are installed and what kind of work they add: tracing, interventions, attribution, sparse features, or visualization. They are not required for the core tabs and are never run automatically.",
        "know": "No prior knowledge is needed. An installation status is not a compatibility guarantee; a compatibility check is a technical environment check, not an analysis of the prompt.",
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
        "know": entry.get("know", ""),
        "examples": list(entry.get("examples", [])),
    }

from __future__ import annotations

import importlib.util


TOOLS = {
    "TransformerLens": "transformer_lens",
    "NNsight": "nnsight",
    "Captum": "captum",
    "UMAP": "umap",
    "SAELens (evaluation candidate)": "sae_lens",
    "Pyvene (evaluation candidate)": "pyvene",
    "BertViz (evaluation candidate)": "bertviz",
    "CircuitsVis (evaluation candidate)": "circuitsvis",
}

TOOL_METADATA = {
    "TransformerLens": {
        "capability": "Hooked model tracing, residual-stream inspection, and interventions.",
        "limitation": "Uses its own model wrapper and may require additional CPU or GPU memory.",
    },
    "NNsight": {
        "capability": "Flexible tracing and interventions on model internals.",
        "limitation": "Model and Transformers-version compatibility must be checked at use time.",
    },
    "Captum": {
        "capability": "Attribution methods such as Integrated Gradients and occlusion.",
        "limitation": "Attribution can require many forwards or backwards passes.",
    },
    "UMAP": {
        "capability": "Nonlinear two-dimensional activation projections.",
        "limitation": "Projection geometry is approximate and sensitive to settings.",
    },
    "SAELens (evaluation candidate)": {
        "capability": "Sparse autoencoder features for activation exploration.",
        "limitation": "No application adapter is enabled; model and feature compatibility are untested.",
    },
    "Pyvene (evaluation candidate)": {
        "capability": "Structured model interventions and representation editing.",
        "limitation": "No application adapter is enabled; architecture support is untested.",
    },
    "BertViz (evaluation candidate)": {
        "capability": "Interactive attention visualizations.",
        "limitation": "No application adapter is enabled; embedding it would add frontend dependencies.",
    },
    "CircuitsVis (evaluation candidate)": {
        "capability": "Interactive circuit and attention-oriented visualizations.",
        "limitation": "No application adapter is enabled; integration and model support are untested.",
    },
}


def installed_tools() -> dict[str, bool]:
    return {
        display_name: importlib.util.find_spec(module_name) is not None
        for display_name, module_name in TOOLS.items()
    }


def tool_status_rows(compatibility: dict[str, str] | None = None) -> list[dict[str, str | bool]]:
    """Return user-facing optional-tool status without importing tools eagerly."""
    status = installed_tools()
    compatibility = compatibility or {}
    return [
        {
            "tool": tool,
            "installed": status[tool],
            "capability": TOOL_METADATA[tool]["capability"],
            "limitation": TOOL_METADATA[tool]["limitation"],
            "compatibility": compatibility.get(tool, "Not checked"),
        }
        for tool in TOOLS
    ]


def probe_tool(tool_name: str, model_name: str) -> str:
    """Run an explicit, isolated compatibility/status probe for one tool."""
    if tool_name == "TransformerLens":
        return transformer_lens_probe(model_name)
    if tool_name == "NNsight":
        return nnsight_status()
    if tool_name in {"Captum", "UMAP"}:
        module_name = TOOLS[tool_name]
        if not installed_tools()[tool_name]:
            return f"{tool_name} is not installed. Install the corresponding optional extra."
        return f"Installed. {tool_name} capability is available; model compatibility is checked when the feature runs."
    if tool_name in TOOLS:
        return "Evaluation candidate only. No application adapter is enabled or executed by default."
    return f"Unknown optional tool: {tool_name}."


def transformer_lens_probe(model_name: str) -> str:
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        return "TransformerLens is not installed."
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device="cpu",
        )
        return (
            f"Supported: {model.cfg.n_layers} layers, "
            f"d_model={model.cfg.d_model}, {model.cfg.n_heads} heads."
        )
    except Exception as exc:
        return f"Installed, but this model could not be loaded: {type(exc).__name__}: {exc}"


def nnsight_status() -> str:
    try:
        import nnsight
    except ImportError:
        return "NNsight is not installed."
    return f"NNsight {getattr(nnsight, '__version__', 'unknown')} is available."

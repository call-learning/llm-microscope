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
        "package": "transformer_lens",
        "capability": "Hooked model tracing, residual-stream inspection, and interventions.",
        "limitation": "Uses its own model wrapper and may require additional CPU or GPU memory.",
        "adapter_enabled": False,
        "visualization_scope": "Tracing and residual-stream inspection",
    },
    "NNsight": {
        "package": "nnsight",
        "capability": "Flexible tracing and interventions on model internals.",
        "limitation": "Model and Transformers-version compatibility must be checked at use time.",
        "adapter_enabled": False,
        "visualization_scope": "Tracing and interventions",
    },
    "Captum": {
        "package": "captum",
        "capability": "Attribution methods such as Integrated Gradients and occlusion.",
        "limitation": "Attribution can require many forwards or backwards passes.",
        "adapter_enabled": True,
        "visualization_scope": "Attribution",
    },
    "UMAP": {
        "package": "umap",
        "capability": "Nonlinear two-dimensional activation projections.",
        "limitation": "Projection geometry is approximate and sensitive to settings.",
        "adapter_enabled": True,
        "visualization_scope": "Activation projection",
    },
    "SAELens (evaluation candidate)": {
        "package": "sae_lens",
        "capability": "Sparse autoencoder features for activation exploration.",
        "limitation": "No application adapter is enabled; model and feature compatibility are untested.",
        "adapter_enabled": False,
        "visualization_scope": "Sparse-feature visualisation",
    },
    "Pyvene (evaluation candidate)": {
        "package": "pyvene",
        "capability": "Structured model interventions and representation editing.",
        "limitation": "No application adapter is enabled; architecture support is untested.",
        "adapter_enabled": False,
        "visualization_scope": "Intervention and representation visualisation",
    },
    "BertViz (evaluation candidate)": {
        "package": "bertviz",
        "capability": "Interactive attention visualizations.",
        "limitation": "No application adapter is enabled; embedding it would add frontend dependencies.",
        "adapter_enabled": False,
        "visualization_scope": "Attention visualisation",
    },
    "CircuitsVis (evaluation candidate)": {
        "package": "circuitsvis",
        "capability": "Interactive circuit and attention-oriented visualizations.",
        "limitation": "No application adapter is enabled; integration and model support are untested.",
        "adapter_enabled": False,
        "visualization_scope": "Circuit and attention visualisation",
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
            "package": TOOL_METADATA[tool]["package"],
            "installed": status[tool],
            "compatibility_checked": tool in compatibility,
            "integration_status": (
                "Integrated" if TOOL_METADATA[tool]["adapter_enabled"] else "Evaluation candidate"
            ),
            "adapter_enabled": TOOL_METADATA[tool]["adapter_enabled"],
            "visualization_scope": TOOL_METADATA[tool]["visualization_scope"],
            "capability": TOOL_METADATA[tool]["capability"],
            "limitation": TOOL_METADATA[tool]["limitation"],
            "compatibility": compatibility.get(tool, "Not checked"),
        }
        for tool in TOOLS
    ]


def probe_tool(tool_name: str, model_name: str) -> str:
    """Run an isolated compatibility/status probe for one optional tool."""
    try:
        return _probe_tool(tool_name, model_name)
    except Exception as exc:
        return (
            f"{tool_name} is unavailable: {type(exc).__name__}: {exc}. "
            "The core visualisations remain available."
        )


def _probe_tool(tool_name: str, model_name: str) -> str:
    """Run the provider-specific part of an optional-tool probe."""
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
        module_name = TOOL_METADATA[tool_name]["package"]
        if not installed_tools()[tool_name]:
            return f"{tool_name} is not installed ({module_name}). No application adapter is enabled."
        return (
            f"{tool_name} is installed, but no application adapter is enabled. "
            "The package is not executed by default."
        )
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

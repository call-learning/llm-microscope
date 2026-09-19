from __future__ import annotations

import importlib.util


TOOLS = {
    "TransformerLens": "transformer_lens",
    "NNsight": "nnsight",
    "Captum": "captum",
    "UMAP": "umap",
}


def installed_tools() -> dict[str, bool]:
    return {
        display_name: importlib.util.find_spec(module_name) is not None
        for display_name, module_name in TOOLS.items()
    }


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


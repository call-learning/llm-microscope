from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from microscope.analysis import (
    analyse,
    cosine_by_layer,
    final_token_evolution,
    layer_norms,
    logit_lens,
    reduce_activations,
    top_tokens,
)
from microscope.interventions import (
    PatchTarget,
    gradient_x_input,
    mean_ablation,
    patch_final_residual,
    patching_score_curve,
)
from microscope.probing import (
    ProbeResult,
    fit_probes,
    generate_concept_labels,
    generate_quick_labels,
)
from microscope.docs import APP_SUBTITLE, APP_TITLE, DEFAULT_PROMPT, SIDEBAR_MODEL_CAPTION, view
from microscope.runtime import cuda_stats, layer_modules, load_runtime
from microscope.toolbox import installed_tools, nnsight_status, transformer_lens_probe


st.set_page_config(page_title="LLM Microscope", page_icon="🔬", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)


@st.cache_resource(show_spinner="Loading model…")
def cached_runtime(model_name: str, trust_remote_code: bool, attn_implementation: str):
    return load_runtime(
        model_name,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )


def _example_button(label: str, set_map: dict, run: str | None) -> bool:
    """Render a suggested-action button and return whether it was just clicked.

    Pre-filling the page's widgets is done by assigning to the widget's
    ``st.session_state[<key>]`` from *this button's own callback* — the
    Streamlit-recommended way to change a widget value without raising the
    "created with a default value but also set via Session State" warning
    (assigning to ``session_state`` for a keyed widget from a *different*
    widget's ``on_click`` is what triggers that warning). ``set_map`` holds the
    widget keys to pre-fill; ``run`` is an optional one-shot trigger name the
    page checks to fire the relevant analysis.
    """

    def _callback() -> None:
        for key, value in set_map.items():
            st.session_state[key] = value

    clicked = st.button(label, key=f"ex_{label}", on_click=_callback)
    if clicked and run:
        st.session_state[f"trigger_{run}"] = True
    return clicked


def page_intro(
    what: str,
    why: str,
    examples: list[dict] | None = None,
    example_label: str = "Suggested actions",
) -> None:
    """Render a compact explainer for a page.

    ``what`` / ``why`` are shown inside a collapsed "About this view"
    expander so they do not crowd the page. ``examples`` is a list of
    ``{"label": str, "set": dict[str, Any], "run": str | None}`` rendered as
    buttons inside a popover: clicking one pre-fills the page's widgets (via an
    ``on_click`` callback that writes session state) and, when ``run`` is
    given, sets a trigger flag the page uses to fire the analysis.
    """
    with st.expander("About this view"):
        st.markdown(f"**What:** {what}\n\n**Why it matters:** {why}")

    if examples:
        with st.popover(example_label, icon=":material/science:"):
            for ex in examples:
                _example_button(ex["label"], ex.get("set", {}), ex.get("run"))


def show_intro(tab_name: str) -> None:
    """Render the explainer for a tab using its docs entry from docs.py."""
    doc = view(tab_name)
    if not doc["what"] and not doc["examples"]:
        return
    page_intro(doc["what"], doc["why"], doc.get("examples") or None)


with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Hugging Face model", "Qwen/Qwen3-1.7B")
    trust_remote_code = st.checkbox("Trust remote model code", value=False)
    max_length = st.slider("Maximum prompt tokens", 16, 512, 128, 16)
    attn_impl = st.radio(
        "Attention kernel",
        ["eager", "sdpa", "flash_attention_2"],
        index=0,
        horizontal=True,
        help="eager exposes the attention weights needed by the Attention view. sdpa / flash_attention_2 are faster but do not return attention matrices.",
    )
    st.caption(SIDEBAR_MODEL_CAPTION)

try:
    runtime = cached_runtime(model_name, trust_remote_code, attn_impl)
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {runtime.model_name}")
    st.write(f"Layers: {len(runtime.layers)}")
    st.write(f"dtype: {runtime.dtype}")
    st.write(f"Attention: {runtime.attn_implementation}")
    for key, value in cuda_stats().items():
        st.write(f"{key}: {value}")

# Pre-seed the prompt value before the widget is created. Using setdefault
# (instead of a literal default on the widget) keeps the field from being blank
# on first load while avoiding Streamlit's "created with a default value but
# also set via Session State" warning: once the user types or a suggested-action
# button sets the value, setdefault is a no-op on later reruns.
st.session_state.setdefault("prompt", DEFAULT_PROMPT)
prompt = st.text_area(
    "Prompt",
    value=None,
    height=90,
    key="prompt",
    help="Type a prompt, or pick one from the suggested actions in each tab's popover.",
)
# Guard against a zero-token input (empty prompt tokenizes to 0 tokens and
# crashes the model's attention reshape).
if not prompt or not prompt.strip():
    prompt = DEFAULT_PROMPT
with_attention = st.checkbox("Capture attention maps", value=True)
run = st.button("Analyse", type="primary") or bool(st.session_state.pop("trigger_analyse", False))

if run:
    with st.spinner("Running analysis…"):
        try:
            st.session_state["analysis"] = analyse(
                runtime,
                prompt,
                max_length=max_length,
                with_attention=with_attention,
            )
            st.session_state["analysis_prompt"] = prompt
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            st.error("CUDA ran out of memory. Shorten the prompt, disable attention maps, or use a smaller model.")
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("analysis")

tabs = st.tabs(
    [
        "Tokens & prediction",
        "Logit lens",
        "Attention",
        "Activations",
        "Compare A/B",
        "Patching",
        "Attribution",
        "Linear Probe",
        "Toolbox",
    ]
)

with tabs[0]:
    show_intro("Tokens & prediction")
    if result is None:
        st.info("Run an analysis first.")
    else:
        token_df = pd.DataFrame(
            {
                "position": range(len(result.labels)),
                "token": result.labels,
                "token_id": result.input_ids[0].tolist(),
            }
        )
        left, right = st.columns(2)
        left.subheader("Tokenisation")
        left.dataframe(token_df, width="stretch", hide_index=True)
        right.subheader("Final next-token prediction")
        right.dataframe(
            top_tokens(runtime, result.logits[0, -1], k=20),
            width="stretch",
            hide_index=True,
        )

with tabs[1]:
    if result is None:
        st.info("Run an analysis first.")
    else:
        show_intro("Logit lens")
        st.subheader("Prediction decoded after every layer")
        st.caption("This applies the model's final normalization and output head to each intermediate residual state.")
        lens = logit_lens(runtime, result)
        st.dataframe(lens, width="stretch", hide_index=True)
        evolution = final_token_evolution(runtime, result)
        st.write(f"Final preferred token: `{evolution.attrs['token']}`")
        st.plotly_chart(
            px.line(evolution, x="layer", y="probability", markers=True),
            width="stretch",
        )

with tabs[2]:
    show_intro("Attention")
    if result is None:
        st.info("Run an analysis first.")
    elif not result.attentions:
        st.warning(
            "No attention matrices were returned. The model is using an optimized attention "
            "kernel (SDPA or FlashAttention), which does not expose attention weights. Switch the "
            "sidebar **Attention kernel** back to **eager** and re-run the analysis."
        )
    else:
        layer = st.slider("Attention layer", 0, len(result.attentions) - 1, 0)
        attention = result.attentions[layer][0]
        head = st.slider("Attention head", 0, attention.shape[0] - 1, 0)
        fig = px.imshow(
            attention[head].numpy(),
            x=result.labels,
            y=result.labels,
            labels={"x": "Key token", "y": "Query token", "color": "Attention"},
            aspect="auto",
            title=f"Layer {layer}, head {head}",
        )
        st.plotly_chart(fig, width="stretch")

with tabs[3]:
    show_intro("Activations")
    if result is None:
        st.info("Run an analysis first.")
    else:
        norms = layer_norms(result)
        st.plotly_chart(px.line(norms, x="layer", y="norm", markers=True), width="stretch")
        layer = st.slider("Projection layer", 0, len(result.hidden_states) - 2, len(result.hidden_states) // 2)
        method = st.radio("Projection", ["PCA", "UMAP"], horizontal=True)
        try:
            projection = reduce_activations(result, layer, method)
            st.plotly_chart(
                px.scatter(projection, x="x", y="y", text="token", hover_data=["position"]),
                width="stretch",
            )
        except Exception as exc:
            st.warning(str(exc))

with tabs[4]:
    show_intro("Compare A/B")
    st.subheader("Compare final-token representations")
    prompt_b = st.text_area("Prompt B", "The capital of Germany is", key="prompt_b")
    if st.button("Compare prompts") or bool(st.session_state.pop("trigger_compare", False)):
        try:
            a = analyse(runtime, prompt, max_length=max_length, with_attention=False)
            b = analyse(runtime, prompt_b, max_length=max_length, with_attention=False)
            comparison = cosine_by_layer(a, b)
            st.plotly_chart(
                px.line(comparison, x="layer", y="cosine_similarity", markers=True),
                width="stretch",
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Comparison failed: {type(exc).__name__}: {exc}")

with tabs[5]:
    show_intro("Patching")
    st.subheader("Causal patching & ablation")
    st.caption("Copies or ablates the final-token state at a layer. Residual patching works on all architectures; attention/MLP output patching requires the self_attn/mlp convention.")

    # Check module availability
    first_layer = runtime.layers[0]
    attn_mod, mlp_mod = layer_modules(first_layer)
    modules_available = attn_mod is not None and mlp_mod is not None

    target_options = ["Residual"]
    if modules_available:
        target_options += ["Attention output", "MLP output"]
    target_label = st.selectbox("Patch target", target_options, key="patch_target_kind")
    target_map = {
        "Residual": PatchTarget.RESIDUAL,
        "Attention output": PatchTarget.ATTN_OUTPUT,
        "MLP output": PatchTarget.MLP_OUTPUT,
    }
    target = target_map[target_label]

    if not modules_available and target_label != "Residual":
        st.warning("Module-level patching is unavailable for this architecture. Falling back to residual.")
        target = PatchTarget.RESIDUAL

    source = st.text_area("Source prompt", "The capital of France is", key="patch_source")
    target_prompt = st.text_area("Target prompt", "The capital of Germany is", key="patch_target")

    # Head selector for attention output
    head_idx = 0
    if target is PatchTarget.ATTN_OUTPUT and modules_available:
        # Determine n_heads from the model config
        cfg = runtime.model.config
        n_heads = getattr(cfg, "num_attention_heads", 16)
        head_idx = st.slider("Query head", 0, n_heads - 1, 0)

    # Mode: single-layer or score curve
    mode = st.radio("Mode", ["Single layer", "Score curve (all layers)"], horizontal=True, key="mode")

    if mode == "Single layer":
        patch_layer = st.slider("Layer to patch", 0, len(runtime.layers) - 1, len(runtime.layers) // 2)

        # Ablation mode
        ablation_mode = st.radio("Ablation", ["Off (patch)", "Zero", "Mean"], horizontal=True, key="ablation")

        run_patch = st.button("Run patching experiment") or bool(st.session_state.pop("trigger_patch_single", False))
        if run_patch:
            try:
                if ablation_mode == "Off (patch)":
                    if target is PatchTarget.RESIDUAL:
                        baseline, patched = patch_final_residual(
                            runtime, source, target_prompt, patch_layer, max_length=max_length,
                        )
                    else:
                        # Use score curve for single layer with non-residual target
                        curve = patching_score_curve(
                            runtime, source, target_prompt, target,
                            max_length=max_length, head_idx=head_idx,
                        )
                        row = curve[curve["layer"] == patch_layer].iloc[0]
                        baseline_df = pd.DataFrame([{
                            "rank": 1,
                            "token": repr(row["baseline_top1_token"]),
                            "token_id": -1,
                            "probability": row["baseline_top1_prob"],
                        }])
                        patched_df = pd.DataFrame([{
                            "rank": 1,
                            "token": repr(row["patched_top1_token"]),
                            "token_id": -1,
                            "probability": row["patched_top1_prob"],
                        }])
                else:
                    ab_mode = "zero" if ablation_mode == "Zero" else "mean"
                    baseline, patched = mean_ablation(
                        runtime, source, target_prompt, patch_layer,
                        target, mode=ab_mode, max_length=max_length,
                    )
                left, right = st.columns(2)
                left.write("Baseline target")
                left.dataframe(baseline, width="stretch", hide_index=True)
                right.write(f"After {'ablation' if ablation_mode != 'Off (patch)' else 'source → target patch'}")
                right.dataframe(patched, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Patching failed: {type(exc).__name__}: {exc}")

    else:  # Score curve
        curve_metric = st.radio("Curve metric", ["Logit delta", "Probability delta"], horizontal=True)
        run_curve = st.button("Run score curve") or bool(st.session_state.pop("trigger_patch_curve", False))
        if run_curve:
            with st.spinner("Running patching score curve (1 source + 1 baseline + N patched forwards)…"):
                try:
                    curve = patching_score_curve(
                        runtime, source, target_prompt, target,
                        max_length=max_length, head_idx=head_idx,
                    )
                    st.session_state["patch_curve"] = curve
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    st.error("CUDA ran out of memory. Shorten the prompt or use a smaller model.")
                except Exception as exc:
                    st.error(f"Score curve failed: {type(exc).__name__}: {exc}")

        curve = st.session_state.get("patch_curve")
        if curve is not None:
            y_col = "delta_logit" if curve_metric == "Logit delta" else "delta_prob"
            st.plotly_chart(
                px.line(curve, x="layer", y=y_col, markers=True,
                        title=f"Patching score curve ({target_label})"),
                width="stretch",
            )
            st.subheader("Per-layer detail")
            st.dataframe(curve, width="stretch", hide_index=True)

with tabs[6]:
    show_intro("Attribution")
    st.subheader("Gradient × input attribution")
    st.caption("Scores input-token contribution to the final position's selected output logit.")
    target_id_text = st.text_input("Target token ID (blank = model's top prediction)", "", key="attr_target")
    run_attr = st.button("Calculate attribution") or bool(st.session_state.pop("trigger_attribution", False))
    if run_attr:
        try:
            target_id = int(target_id_text) if target_id_text.strip() else None
            attribution, chosen_id = gradient_x_input(
                runtime,
                prompt,
                target_token_id=target_id,
                max_length=max_length,
            )
            chosen = runtime.tokenizer.decode([chosen_id], clean_up_tokenization_spaces=False)
            st.write(f"Target token: `{chosen!r}` (ID {chosen_id})")
            st.plotly_chart(
                px.bar(attribution, x="token", y="importance", hover_data=["position"]),
                width="stretch",
            )
            st.dataframe(attribution, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Attribution failed: {type(exc).__name__}: {exc}")

with tabs[7]:
    show_intro("Linear Probe")
    st.subheader("Linear Probe")
    st.caption("Fits a binary L2 logistic regression on the final-token hidden state at each layer. Shows where a property becomes linearly decodable.")

    probe_mode = st.radio("Label mode", ["Concept (contrastive)", "Quick (substring)"], horizontal=True, key="probe_mode")

    prompts: list[str] = []
    labels: list[int] = []

    if probe_mode == "Concept (contrastive)":
        pos_template = st.text_input("Positive template", "The capital of {X} is", key="pos_template")
        neg_template = st.text_input("Negative template", "The largest city in {X} is", key="neg_template")
        fill_values = st.text_input("Fill values (comma-separated)", "France, Germany, Japan, Spain, Italy, Brazil", key="fill_values")
        run_probe_concept = st.button("Fit probe (concept)") or bool(st.session_state.pop("trigger_probe_concept", False))
        if run_probe_concept:
            try:
                prompts, labels = generate_concept_labels(pos_template, neg_template, fill_values)
                with st.spinner(f"Fitting probe on {len(prompts)} prompts…"):
                    probe_result: ProbeResult = fit_probes(runtime, prompts, labels, C=st.session_state.get("probe_C", 1.0))
                st.session_state["probe_result"] = probe_result
            except Exception as exc:
                st.error(f"Probe failed: {type(exc).__name__}: {exc}")
    else:
        substring = st.text_input("Property substring", "France", key="substring")
        quick_prompts = st.text_area(
            "Prompts (one per line)",
            "The capital of France is\nThe capital of Germany is\nThe capital of Japan is\nThe largest city in France is\nThe largest city in Germany is\nThe largest city in Japan is",
            height=150,
            key="quick_prompts",
        )
        run_probe_quick = st.button("Fit probe (quick)") or bool(st.session_state.pop("trigger_probe_quick", False))
        if run_probe_quick:
            try:
                prompts = [p.strip() for p in quick_prompts.split("\n") if p.strip()]
                labels = generate_quick_labels(prompts, substring)
                if len(set(labels)) < 2:
                    st.warning("Need at least one prompt with and one without the substring.")
                else:
                    with st.spinner(f"Fitting probe on {len(prompts)} prompts…"):
                        probe_result = fit_probes(runtime, prompts, labels, C=st.session_state.get("probe_C", 1.0))
                    st.session_state["probe_result"] = probe_result
            except Exception as exc:
                st.error(f"Probe failed: {type(exc).__name__}: {exc}")

    C_options = list(np.unique(np.logspace(-2, 2, 41)))
    C_slider = st.select_slider(
        "Penalty strength C (higher = less regularization)",
        options=[round(float(c), 4) for c in C_options],
        value=1.0,
        format_func=lambda c: f"{c:g}",
    )
    st.session_state["probe_C"] = float(C_slider)

    probe = st.session_state.get("probe_result")
    if probe is not None:
        st.write(f"Training set: {probe.n_prompts} prompts ({probe.n_positive} pos, {probe.n_negative} neg)")
        st.plotly_chart(
            px.line(probe.per_layer_accuracy, x="layer", y="accuracy", markers=True,
                    title="Probe accuracy per layer"),
            width="stretch",
        )

        best_layer = int(probe.per_layer_accuracy.loc[probe.per_layer_accuracy["accuracy"].idxmax(), "layer"])
        direction = probe.direction[best_layer]
        top_idx = np.argsort(np.abs(direction))[::-1][:10]
        dir_df = pd.DataFrame({
            "index": top_idx,
            "value": [direction[i] for i in top_idx],
        })
        st.subheader(f"Probe direction at layer {best_layer} (top-10 components)")
        st.dataframe(dir_df, width="stretch", hide_index=True)

        if st.button("Copy direction vector"):
            st.clipboard.set_text(repr(direction))
            st.success("Direction vector copied to clipboard.")

with tabs[8]:
    show_intro("Toolbox")
    st.subheader("Optional tool integrations")
    status = installed_tools()
    st.dataframe(
        pd.DataFrame([{"tool": tool, "installed": installed} for tool, installed in status.items()]),
        width="stretch",
        hide_index=True,
    )
    st.write(nnsight_status())
    st.caption("TransformerLens uses its own model wrapper and can require additional VRAM. The probe deliberately loads on CPU.")
    if st.button("Probe TransformerLens compatibility"):
        st.write(transformer_lens_probe(model_name))


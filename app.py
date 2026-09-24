from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import torch

from microscope.analysis import (
    analyse,
    cosine_by_layer,
    attention_head_similarity,
    attention_head_summary,
    attention_rollout,
    final_token_evolution,
    layer_norms,
    layer_prediction_metrics,
    logit_lens,
    pairwise_cosine_distances,
    representation_metrics,
    reduce_activations,
    top_tokens,
)
from microscope.interventions import (
    PatchTarget,
    gradient_x_input,
    mean_ablation,
    patch_final_residual,
    patching_effect_matrix,
    patching_score_curve,
    integrated_gradients,
)
from microscope.probing import (
    ProbeResult,
    fit_probes,
    generate_concept_labels,
    generate_quick_labels,
)
from microscope.docs import (
    APP_GUIDE,
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_PROMPT,
    SIDEBAR_MODEL_CAPTION,
    view,
)
from microscope.runtime import cuda_stats, layer_modules, load_runtime
from microscope.toolbox import probe_tool, tool_status_rows


st.set_page_config(page_title="LLM Microscope", page_icon="🔬", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
with st.expander("How to use this workbench", expanded=True):
    st.markdown(APP_GUIDE)


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
    know: str = "",
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
        st.markdown(f"**What it is:** {what}\n\n**What it helps you understand:** {why}")
        if know:
            st.markdown(f"**Prior knowledge:** {know}")

    if examples:
        with st.popover(example_label, icon=":material/science:"):
            for ex in examples:
                _example_button(ex["label"], ex.get("set", {}), ex.get("run"))


def show_intro(tab_name: str) -> None:
    """Render the explainer for a tab using its docs entry from docs.py."""
    doc = view(tab_name)
    if not doc["what"] and not doc["examples"]:
        return
    page_intro(doc["what"], doc["why"], doc.get("know", ""), doc.get("examples") or None)


def analysis_identity(model_name: str, prompt: str, max_length: int) -> tuple[str, str, int]:
    """Identify derived visual data belonging to the current analysis inputs."""
    return model_name, prompt, max_length


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
        help="How the attention math is executed. 'eager' is the reference implementation and returns the attention weights, so the Attention tab needs it. 'sdpa' and 'flash_attention_2' are faster but only return the final hidden states, not the per-head attention matrices. Leave it on 'eager' unless you are only using the other views.",
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
    help="The text sent to the model. Decoder-only language models read the prompt from left to right and predict what token should come next. You can use a sentence, a question, or code. The model's tokenizer may split familiar words into several tokens; inspect Tokens & prediction to see the actual input.",
)
# Guard against a zero-token input (empty prompt tokenizes to 0 tokens and
# crashes the model's attention reshape).
if not prompt or not prompt.strip():
    prompt = DEFAULT_PROMPT
with_attention = st.checkbox(
    "Capture attention maps",
    value=True,
    help="When enabled, the model returns the per-layer, per-head attention weights used by the Attention tab. This uses extra memory, and optimized attention kernels may not expose these weights. Turn it off for faster or lower-memory runs when you do not need that tab.",
)
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
            st.session_state["analysis_identity"] = analysis_identity(
                runtime.model_name, prompt, max_length,
            )
            st.session_state.pop("prediction_metrics_cache", None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            st.error("CUDA ran out of memory. Shorten the prompt, disable attention maps, or use a smaller model.")
        except Exception as exc:
            st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("analysis")
current_analysis = st.session_state.get("analysis_identity") == analysis_identity(
    runtime.model_name, prompt, max_length,
)

tabs = st.tabs(
    [
        "Tokens & prediction",
        "Prediction journey",
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
    show_intro("Prediction journey")
    if result is None or not current_analysis:
        st.info("Run an analysis for the current prompt first.")
    else:
        candidate_df = top_tokens(runtime, result.logits[0, -1], k=20)
        candidate_ids = candidate_df["token_id"].tolist()
        selected_id = st.selectbox(
            "Candidate token to follow",
            candidate_ids,
            format_func=lambda token_id: (
                f"{runtime.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)!r} "
                f"(ID {token_id})"
            ),
            key="journey_token_id",
        )
        journey_metric = st.radio(
            "Journey metric",
            ["Probability", "Rank", "Logit", "Entropy"],
            horizontal=True,
            key="journey_metric",
        )
        metric_column = journey_metric.lower()
        try:
            cache_key = (current_analysis, int(selected_id))
            cache = st.session_state.setdefault("prediction_metrics_cache", {})
            if cache_key not in cache:
                with st.spinner("Decoding the selected token through all layers…"):
                    cache[cache_key] = layer_prediction_metrics(runtime, result, int(selected_id))
            journey = cache[cache_key]
            position = st.slider(
                "Prompt position for the layer chart",
                0,
                len(result.labels) - 1,
                len(result.labels) - 1,
                format="%d",
            )
            heatmap = journey.pivot(index="layer", columns="position", values=metric_column)
            st.plotly_chart(
                px.imshow(
                    heatmap,
                    aspect="auto",
                    labels={"x": "Prompt token position", "y": "Layer", "color": journey_metric},
                    x=[f"{index}: {label.split(': ', 1)[-1]}" for index, label in enumerate(result.labels)],
                    title=f"{journey_metric} for {runtime.tokenizer.decode([int(selected_id)], clean_up_tokenization_spaces=False)!r}",
                ),
                width="stretch",
            )
            position_curve = journey[journey["position"] == position]
            st.plotly_chart(
                px.line(
                    position_curve,
                    x="layer",
                    y=metric_column,
                    markers=True,
                    title=f"{journey_metric} at position {position}: {result.labels[position]}",
                ),
                width="stretch",
            )
            st.caption(
                "Entropy measures how concentrated the full next-token distribution is at a position. "
                "Higher entropy means less concentration, not necessarily an incorrect prediction."
            )
            left, right = st.columns(2)
            left.subheader("Final-position candidates")
            left.plotly_chart(
                px.bar(
                    candidate_df.sort_values("probability"),
                    x="probability",
                    y="token",
                    orientation="h",
                    hover_data=["token_id", "rank"],
                    labels={"probability": "Probability", "token": "Candidate token"},
                ),
                width="stretch",
            )
            right.subheader("Final-layer entropy by position")
            final_entropy = journey[journey["layer"] == journey["layer"].max()]
            right.plotly_chart(
                px.line(
                    final_entropy,
                    x="position",
                    y="entropy",
                    markers=True,
                    hover_data=["token"],
                    labels={"entropy": "Entropy", "position": "Prompt token position"},
                ),
                width="stretch",
            )
        except Exception as exc:
            st.warning(f"Prediction journey unavailable: {type(exc).__name__}: {exc}")

with tabs[2]:
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

with tabs[3]:
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
        token_axes = [f"{position}: {label.split(': ', 1)[-1]}" for position, label in enumerate(result.labels)]
        fig = px.imshow(
            attention[head].numpy(),
            x=token_axes,
            y=token_axes,
            labels={"x": "Key token", "y": "Query token", "color": "Attention"},
            aspect="auto",
            title=f"Layer {layer}, head {head}",
        )
        st.plotly_chart(fig, width="stretch")

        summary = attention_head_summary(result.attentions, result.labels)
        selected_summary = summary[
            (summary["layer"] == layer) & (summary["head"] == head)
        ]
        st.subheader("Selected-head summary")
        st.dataframe(
            selected_summary[
                ["query_position", "query_token", "entropy", "max_key_position", "max_key_token", "max_attention"]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Attention entropy measures how concentrated a query row is over key positions. "
            "It is not a measure of causal importance."
        )

        compare_heads = st.multiselect(
            "Heads to compare",
            list(range(attention.shape[0])),
            default=[head],
            format_func=lambda value: f"Head {value}",
            key="attention_compare_heads",
        )
        if compare_heads:
            comparison = summary[
                (summary["layer"] == layer) & summary["head"].isin(compare_heads)
            ]
            st.plotly_chart(
                px.line(
                    comparison,
                    x="query_position",
                    y="entropy",
                    color="head",
                    markers=True,
                    hover_data=["query_token", "max_key_position", "max_key_token"],
                    labels={"query_position": "Query position", "entropy": "Attention entropy"},
                    title=f"Head comparison at layer {layer}",
                ),
                width="stretch",
            )
        if st.checkbox("Show raw-pattern head similarity", key="attention_similarity"):
            similarity = attention_head_similarity(result.attentions, layer)
            st.plotly_chart(
                px.imshow(
                    similarity,
                    aspect="auto",
                    labels={"x": "Head", "y": "Head", "color": "Cosine similarity"},
                    title=f"Raw attention-pattern similarity at layer {layer}",
                ),
                width="stretch",
            )

        if st.checkbox("Show attention rollout", key="attention_rollout"):
            try:
                rollout = attention_rollout(result.attentions, include_residual=True)
                st.plotly_chart(
                    px.imshow(
                        rollout.numpy(),
                        x=token_axes,
                        y=token_axes,
                        labels={"x": "Key token", "y": "Query token", "color": "Rollout weight"},
                        aspect="auto",
                        title="Attention rollout across layers and heads",
                    ),
                    width="stretch",
                )
                st.caption(
                    "Rollout averages heads, adds an identity residual connection, row-normalizes each "
                    "layer, and composes layers from early to late. It is descriptive, not automatically causal."
                )
            except Exception as exc:
                st.warning(f"Attention rollout unavailable: {type(exc).__name__}: {exc}")

with tabs[4]:
    show_intro("Activations")
    if result is None:
        st.info("Run an analysis first.")
    else:
        norms = layer_norms(result)
        st.plotly_chart(px.line(norms, x="layer", y="norm", markers=True), width="stretch")
        activation_metric_label = st.radio(
            "Activation heatmap metric",
            ["Hidden-state norm", "Layer-to-layer cosine change"],
            horizontal=True,
            key="activation_metric",
        )
        activation_metric = "norm" if activation_metric_label == "Hidden-state norm" else "cosine_change"
        activation_data = representation_metrics(result, activation_metric)
        activation_heatmap = activation_data.pivot(index="layer", columns="position", values="value")
        st.plotly_chart(
            px.imshow(
                activation_heatmap,
                aspect="auto",
                labels={"x": "Prompt token position", "y": "Layer", "color": activation_metric_label},
                x=[f"{index}: {label.split(': ', 1)[-1]}" for index, label in enumerate(result.labels)],
                title=activation_metric_label,
            ),
            width="stretch",
        )
        token_position = st.slider(
            "Token position for representation trajectory",
            0,
            len(result.labels) - 1,
            len(result.labels) - 1,
            key="activation_token_position",
        )
        trajectory = activation_data[activation_data["position"] == token_position]
        st.plotly_chart(
            px.line(
                trajectory,
                x="layer",
                y="value",
                markers=True,
                hover_data=["token", "position"],
                labels={"value": activation_metric_label},
                title=f"{activation_metric_label} trajectory for {result.labels[token_position]}",
            ),
            width="stretch",
        )
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
        if st.checkbox("Show pairwise cosine-distance matrix", key="activation_pairwise"):
            try:
                distances = pairwise_cosine_distances(result, layer)
                st.plotly_chart(
                    px.imshow(
                        distances,
                        aspect="auto",
                        labels={"x": "Token", "y": "Token", "color": "Cosine distance"},
                        title=f"Pairwise cosine distance at layer {layer}",
                    ),
                    width="stretch",
                )
            except Exception as exc:
                st.warning(f"Pairwise comparison unavailable: {type(exc).__name__}: {exc}")

with tabs[5]:
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

with tabs[6]:
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
                    patch_target_key = target.value
                    patch_target_display = target_label
                    if target is PatchTarget.ATTN_OUTPUT:
                        patch_target_display = f"{target_label} (head {head_idx})"
                    patch_runs = st.session_state.setdefault("patch_curve_runs", {})
                    run_key = (
                        runtime.model_name,
                        source,
                        target_prompt,
                        max_length,
                        patch_target_key,
                        head_idx,
                    )
                    patch_runs[run_key] = {
                        "curve": curve,
                        "model": runtime.model_name,
                        "source": source,
                        "target": target_prompt,
                        "max_length": max_length,
                        "patch_target": patch_target_key,
                        "patch_target_display": patch_target_display,
                        "head_idx": head_idx,
                        "metrics": ["delta_logit", "delta_prob"],
                        "display_metric": curve_metric,
                    }
                    st.session_state["patch_curve_meta"] = patch_runs[run_key]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    st.error("CUDA ran out of memory. Shorten the prompt or use a smaller model.")
                except Exception as exc:
                    st.error(f"Score curve failed: {type(exc).__name__}: {exc}")

        curve = st.session_state.get("patch_curve")
        if curve is not None:
            y_col = "delta_logit" if curve_metric == "Logit delta" else "delta_prob"
            metadata = st.session_state.get("patch_curve_meta", {})
            st.caption(
                f"Source: {metadata.get('source', source)!r} | "
                f"Target: {metadata.get('target', target_prompt)!r} | "
                f"Baseline: unmodified target | Metric: {curve_metric}"
            )
            st.plotly_chart(
                px.line(curve, x="layer", y=y_col, markers=True,
                        title=f"Patching score curve ({target_label})"),
                width="stretch",
            )
            st.subheader("Per-layer detail")
            st.dataframe(curve, width="stretch", hide_index=True)
            st.caption(
                "A changed output is evidence that the intervention influenced this prediction; "
                "it is not automatically a human-readable concept attribution."
            )

            compatible_runs = [
                run
                for run in st.session_state.get("patch_curve_runs", {}).values()
                if run["model"] == runtime.model_name
                and run["source"] == source
                and run["target"] == target_prompt
                and run["max_length"] == max_length
                and y_col in run.get("metrics", [])
            ]
            if compatible_runs:
                matrix = patching_effect_matrix(compatible_runs, y_col)
                st.subheader("Patch-target comparison")
                st.caption(
                    "This matrix compares only completed, compatible runs. Missing targets are not zero effects."
                )
                st.plotly_chart(
                    px.imshow(
                        matrix,
                        aspect="auto",
                        labels={"x": "Patch layer", "y": "Patch target", "color": curve_metric},
                        title=f"Patching effect matrix ({curve_metric})",
                    ),
                    width="stretch",
                )

with tabs[7]:
    show_intro("Attribution")
    st.subheader("Token attribution comparison")
    st.caption("Scores input-token contribution to the final position's selected output logit. Scores are signed and local to the selected output token.")
    target_id_text = st.text_input("Target token ID (blank = model's top prediction)", "", key="attr_target")
    attribution_methods = st.multiselect(
        "Attribution methods",
        ["Gradient x Input", "Integrated Gradients"],
        default=["Gradient x Input"],
        key="attribution_methods",
    )
    ig_steps = 16
    if "Integrated Gradients" in attribution_methods:
        ig_steps = st.slider("Integrated Gradients steps", 2, 64, 16, 2)
    run_attr = st.button("Calculate attribution") or bool(st.session_state.pop("trigger_attribution", False))
    if run_attr:
        try:
            if not attribution_methods:
                raise ValueError("Select at least one attribution method.")
            target_id = int(target_id_text) if target_id_text.strip() else None
            results = {}
            method_errors = {}
            chosen_id = target_id
            with st.spinner("Calculating attribution…"):
                for method in attribution_methods:
                    try:
                        if method == "Gradient x Input":
                            attribution, chosen_id = gradient_x_input(
                                runtime, prompt, target_token_id=target_id, max_length=max_length,
                            )
                        else:
                            attribution, chosen_id = integrated_gradients(
                                runtime, prompt, target_token_id=target_id,
                                max_length=max_length, n_steps=ig_steps,
                            )
                        results[method] = attribution
                    except Exception as exc:
                        method_errors[method] = f"{type(exc).__name__}: {exc}"
            st.session_state["attribution_results"] = results
            st.session_state["attribution_target_id"] = chosen_id
            st.session_state["attribution_errors"] = method_errors
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            st.error("CUDA ran out of memory. Reduce the prompt length or Integrated Gradients steps.")
        except Exception as exc:
            st.error(f"Attribution failed: {type(exc).__name__}: {exc}")

    attribution_results = st.session_state.get("attribution_results", {})
    chosen_id = st.session_state.get("attribution_target_id")
    for method, message in st.session_state.get("attribution_errors", {}).items():
        st.warning(f"{method} unavailable: {message}")
    if attribution_results and chosen_id is not None:
        chosen = runtime.tokenizer.decode([chosen_id], clean_up_tokenization_spaces=False)
        st.write(f"Target token: `{chosen!r}` (ID {chosen_id})")
        for method, attribution in attribution_results.items():
            st.subheader(method)
            st.plotly_chart(
                px.bar(
                    attribution,
                    x="token",
                    y="importance",
                    color="importance",
                    color_continuous_scale="RdBu",
                    hover_data=["position", "magnitude"],
                    labels={"importance": "Signed attribution"},
                ),
                width="stretch",
            )
            st.dataframe(attribution, width="stretch", hide_index=True)
        st.caption(
            "Positive values support the selected output token and negative values oppose it. "
            "Attribution is method-dependent and local; agreement between methods is not proof of a complete causal explanation."
        )

with tabs[8]:
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
        st.write(
            f"Training set: {probe.n_prompts} prompts "
            f"({probe.n_positive} positive, {probe.n_negative} negative)"
        )
        if max(probe.n_positive, probe.n_negative) / max(probe.n_prompts, 1) > 0.6:
            st.warning(
                "The probe labels are imbalanced. Accuracy may be dominated by the majority class."
            )
        st.caption(
            "Probe accuracy indicates decodability, not that the model uses the feature for its answer. "
            "Prompt templates can also provide shortcuts."
        )
        if not probe.heldout_supported:
            st.warning(
                "This dataset is too small for a meaningful stratified held-out split. "
                "The accuracy curve is training-only."
            )
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=probe.per_layer_accuracy["layer"],
            y=probe.per_layer_accuracy["train_accuracy"],
            mode="lines+markers",
            name="Training accuracy",
        ))
        if probe.heldout_supported:
            figure.add_trace(go.Scatter(
                x=probe.per_layer_accuracy["layer"],
                y=probe.per_layer_accuracy["heldout_accuracy"],
                error_y={"type": "data", "array": probe.per_layer_accuracy["heldout_std"].fillna(0)},
                mode="lines+markers",
                name="Held-out accuracy",
            ))
        figure.add_trace(go.Scatter(
            x=probe.per_layer_accuracy["layer"],
            y=probe.per_layer_accuracy["baseline_accuracy"],
            mode="lines",
            name="Majority baseline",
            line={"dash": "dash"},
        ))
        figure.update_layout(title="Probe accuracy per layer", xaxis_title="Layer", yaxis_title="Accuracy")
        st.plotly_chart(figure, width="stretch")

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

with tabs[9]:
    show_intro("Toolbox")
    st.subheader("Optional tool integrations")
    tool_compatibility = st.session_state.get("tool_compatibility", {})
    st.dataframe(
        pd.DataFrame(tool_status_rows(tool_compatibility)),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Optional tools add tracing, interventions, attribution, sparse features, or visualization. "
        "They are isolated from the core Hugging Face analysis and are never executed automatically. "
        "Installed means the package is importable; Integration status means this app has an adapter; "
        "Compatibility contains the result of an explicit check."
    )
    probe_tool_name = st.selectbox(
        "Tool compatibility/status check",
        [row["tool"] for row in tool_status_rows(tool_compatibility)],
        key="tool_probe_name",
    )
    if st.button("Run selected tool check"):
        with st.spinner(f"Checking {probe_tool_name}…"):
            try:
                result_text = probe_tool(probe_tool_name, model_name)
            except Exception as exc:
                result_text = (
                    f"{probe_tool_name} is unavailable: {type(exc).__name__}: {exc}. "
                    "The core visualisations remain available."
                )
        tool_compatibility[probe_tool_name] = result_text
        st.session_state["tool_compatibility"] = tool_compatibility
        if " is unavailable: " in result_text:
            st.warning(result_text)
        else:
            st.write(result_text)

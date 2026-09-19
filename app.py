from __future__ import annotations

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
from microscope.interventions import gradient_x_input, patch_final_residual
from microscope.runtime import cuda_stats, load_runtime
from microscope.toolbox import installed_tools, nnsight_status, transformer_lens_probe


st.set_page_config(page_title="LLM Microscope", page_icon="🔬", layout="wide")
st.title("🔬 LLM Microscope")
st.caption("Local interpretability workbench — logit lens, attention, activations, attribution and causal patching")


@st.cache_resource(show_spinner="Loading model…")
def cached_runtime(model_name: str, trust_remote_code: bool):
    return load_runtime(model_name, trust_remote_code=trust_remote_code)


with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Hugging Face model", "Qwen/Qwen3-1.7B")
    trust_remote_code = st.checkbox("Trust remote model code", value=False)
    max_length = st.slider("Maximum prompt tokens", 16, 512, 128, 16)
    st.caption("Changing the model reloads it. Start with 1.7B on a 12 GB GPU.")

try:
    runtime = cached_runtime(model_name, trust_remote_code)
except Exception as exc:
    st.error(f"Model loading failed: {type(exc).__name__}: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {runtime.model_name}")
    st.write(f"Layers: {len(runtime.layers)}")
    st.write(f"dtype: {runtime.dtype}")
    for key, value in cuda_stats().items():
        st.write(f"{key}: {value}")

prompt = st.text_area("Prompt", "The capital of France is", height=90)
with_attention = st.checkbox("Capture attention maps", value=True)
run = st.button("Analyse", type="primary")

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
        "Toolbox",
    ]
)

with tabs[0]:
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
    if result is None:
        st.info("Run an analysis first.")
    elif not result.attentions:
        st.warning("No attention matrices were returned. Optimized attention implementations (for example SDPA) do not expose weights; re-run with the model's attention implementation set to eager.")
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
    st.subheader("Compare final-token representations")
    prompt_b = st.text_area("Prompt B", "The capital of Germany is", key="prompt_b")
    if st.button("Compare prompts"):
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
    st.subheader("Causal residual-stream patching")
    st.caption("Copies the final-token state after one source layer into the same layer of the target run.")
    source = st.text_area("Source prompt", "The capital of France is", key="patch_source")
    target = st.text_area("Target prompt", "The capital of Germany is", key="patch_target")
    patch_layer = st.slider("Layer to patch", 0, len(runtime.layers) - 1, len(runtime.layers) // 2)
    if st.button("Run patching experiment"):
        try:
            baseline, patched = patch_final_residual(
                runtime,
                source,
                target,
                patch_layer,
                max_length=max_length,
            )
            left, right = st.columns(2)
            left.write("Baseline target")
            left.dataframe(baseline, width="stretch", hide_index=True)
            right.write("After source → target patch")
            right.dataframe(patched, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Patching failed: {type(exc).__name__}: {exc}")

with tabs[6]:
    st.subheader("Gradient × input attribution")
    st.caption("Scores input-token contribution to the final position's selected output logit.")
    target_id_text = st.text_input("Target token ID (blank = model's top prediction)", "")
    if st.button("Calculate attribution"):
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


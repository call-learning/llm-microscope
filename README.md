# LLM Microscope

A local Streamlit workbench for exploring the internals of decoder-only LLMs. The default model is `Qwen/Qwen3-1.7B`, which is a comfortable fit on a 12 GB RTX 3090.

The working views use Hugging Face directly, so the app remains useful even if an optional research library does not support a particular model release.

## Included views

- token IDs and token text;
- final next-token probabilities;
- layer-by-layer logit lens;
- per-head attention heatmaps;
- hidden-state norms and cosine similarity;
- PCA and optional UMAP projections;
- prompt A/B representation comparison;
- causal residual-stream activation patching;
- gradient × input attribution;
- optional TransformerLens and NNsight probes.

## Quick start

You need Python 3.11 or 3.12 and a CUDA-enabled PyTorch installation. The safest route is to retain the PyTorch installation that already works on your machine.

### Existing CUDA/PyTorch environment

```bash
unzip llm-microscope.zip
cd llm-microscope

python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[captum,umap]'
streamlit run app.py
```

### With `uv`

If `uv` selects the wrong PyTorch build, use the existing-environment method above.

```bash
unzip llm-microscope.zip
cd llm-microscope
uv sync --extra captum --extra umap
uv run streamlit run app.py
```

Open <http://localhost:8501>.

Check the GPU first:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Optional research integrations

Install individually because these libraries evolve quickly and may constrain PyTorch or Transformers versions:

```bash
pip install -e '.[nnsight]'
pip install -e '.[transformer-lens]'
```

Or install everything:

```bash
pip install -e '.[all]'
```

The **Toolbox** page detects installed integrations and runs small smoke probes. The main analysis pages do not depend on NNsight or TransformerLens.

## Suggested first experiments

1. Run `The capital of France is` and inspect **Logit lens**.
2. Compare it with `The capital of Germany is` under **Compare A/B**.
3. In **Patching**, copy one source layer's final-token residual into the target and observe how the target next-token probabilities change.
4. Try `def fibonacci(n):` or a short Moodle/PHP prompt and inspect token attribution and activation clusters.

## Memory notes for a 12 GB RTX 3090

- Start with Qwen3-1.7B and prompts below 128 tokens.
- Attention matrices scale quadratically with prompt length.
- The app runs one analysis at a time and moves cached tensors to CPU.
- Qwen3-4B may fit in half precision for basic views, but attention and gradient attribution leave much less headroom.
- If you hit OOM, restart the Streamlit process after reducing the prompt length or model size.

## Important limitations

- Attention is not an explanation by itself.
- A logit lens decodes intermediate representations through the final normalization and language-model head; the model does not literally make a prediction at every layer.
- Gradient attribution is local and sensitive to the selected output token.
- Activation patching provides causal evidence for the patched state, but does not automatically identify a human-readable concept.
- `trust_remote_code` is disabled by default. Only enable it for a repository you trust.


## Context

The workbench already captures hidden states at every layer via `analyse()` (`output_hidden_states=True`), stored as `AnalysisResult.hidden_states` — a tuple of `(batch, seq, hidden)` float32 tensors on CPU. The final-token hidden state at layer L is `hidden_states[L+1][0, -1, :]` (shape `(hidden,)`).

`sklearn` is already a dependency (`scikit-learn>=1.5`). The `LogisticRegression` estimator is the natural fit for a binary linear probe.

The existing `Runtime` holds the model, tokenizer, device, and dtype. The `analyse()` function already tokenizes, runs the forward, and returns hidden states. No changes to `runtime.py` or `analysis.py` are needed for the probe — it is a pure post-processing step on already-captured hidden states.

## Goals / Non-Goals

**Goals:**
- Fit a binary L2-regularized logistic regression on the final-token hidden state at each decoder layer.
- Report per-layer accuracy.
- Report the learned weight vector (probe direction) for the fitted layer.
- Provide two label modes: contrastive (concept) and quick (substring).
- Both modes feed one shared fitting routine.

**Non-Goals:**
- Steering / activation-addition UI (adding a scaled probe direction to an activation and observing the effect). This is a follow-on change.
- Multi-class probes (the probe is binary: property present / absent).
- Probes on non-final-token positions (the probe targets the final-token hidden state, consistent with the rest of the workbench).
- Cross-validation / statistical significance reporting. The probe reports raw per-layer accuracy; the user judges significance.

## Decisions

### 1. Label generation: concept mode

**Decision:** The user provides a positive template (e.g. `The capital of {X} is`), a negative template (e.g. `The largest city in {X} is`), and a comma-separated list of fill values (e.g. `France, Germany, Japan`). The system fills each template with each fill value to produce N positive prompts (label 1) and N negative prompts (label 0). The `{X}` placeholder is replaced with each fill value.

**Rationale:** This is the standard TDD (Tensor Dissection D) probing setup. The contrastive pair isolates the *concept* (e.g. "this is about a capital") from the surface wording. The fill values provide the variation that prevents the probe from learning a trivial string match.

**Implementation sketch:**
```python
def generate_concept_labels(pos_template, neg_template, fill_values):
    prompts, labels = [], []
    for val in fill_values:
        prompts.append(pos_template.replace("{X}", val.strip()))
        labels.append(1)
        prompts.append(neg_template.replace("{X}", val.strip()))
        labels.append(0)
    return prompts, labels
```

### 2. Label generation: quick mode

**Decision:** The user provides a property substring (e.g. `France`) and a set of prompts (either typed directly or generated from a template with fill values). The system labels each prompt 1 if it contains the substring, 0 otherwise.

**Rationale:** Quick mode is a fast way to check "where does the surface form get encoded" without constructing a contrastive pair. It is less powerful than concept mode but requires less user setup.

**Implementation sketch:**
```python
def generate_quick_labels(prompts, substring):
    labels = [1 if substring in p else 0 for p in prompts]
    return prompts, labels
```

### 3. Shared fitting core

**Decision:** Both label modes produce `(prompts: list[str], labels: list[int])`. A single function `fit_probes(runtime, prompts, labels, penalty)` tokenizes the prompts, runs `analyse()` for each (or batched), extracts the final-token hidden state at each layer, fits `LogisticRegression(C=1/penalty)` per layer, and returns a result struct with per-layer accuracy and the weight vector.

**Rationale:** The only difference between concept and quick mode is the label generator. The fitting logic — tokenize, forward, extract hidden state, fit classifier, report — is identical. Duplicating it would be a maintenance burden and would risk divergence.

**Alternative considered:** Separate fitting functions for each mode. Rejected: the only difference is labels; the fitting logic is identical. A shared core is simpler and less error-prone.

### 4. Batch vs per-prompt forward for probe training

**Decision:** Tokenize all prompts and run them as a **single batched forward** (batch dimension = number of prompts). The `analyse()` function already supports batched input (it passes `**batch` to the model). The hidden states are extracted per-prompt, per-layer: `hidden_states[L+1][i, -1, :]` for prompt i.

**Rationale:** A single batched forward is much faster than N separate forwards. For N=20–50 prompts (typical for a probe), a single batched forward with sequence length ≤ 128 is well within the 12 GB budget. The batch dimension is small enough that memory is not a concern.

**Caveat:** The `analyse()` function currently assumes a single-prompt run in some places (e.g. `token_labels` iterates over `input_ids[0]`). For batched input, we need to either:
- (a) extend `analyse()` to handle batches, or
- (b) write a separate `analyse_batch()` that returns hidden states without the single-prompt assumptions.

**Decision:** Option (b) — a separate `analyse_batch()` in `analysis.py` that returns `(input_ids, hidden_states, logits)` for a batch, without the single-prompt `labels` field. This keeps `analyse()` unchanged and avoids the risk of breaking the existing single-prompt views.

### 5. Penalty exposure

**Decision:** Expose the L2 penalty strength as a slider in the UI. The `LogisticRegression(C=...)` parameter is the inverse of the penalty strength: `C = 1 / penalty`. The slider range is `C ∈ [0.01, 100]` (log scale), which covers under-regularized to heavily regularized.

**Rationale:** The user should be able to sweep regularization if overfitting is suspected (e.g. small N in concept mode). The default is `C=1.0` (moderate regularization).

### 6. Direction reporting

**Decision:** After fitting, the probe's `coef_` attribute is the weight vector (shape `(hidden,)` for a binary classifier). We report this vector as the "probe direction" for the fitted layer. The UI shows the top-10 largest-magnitude components of the direction vector (with their indices) as a quick sanity check, and makes the full vector available for copy/export.

**Rationale:** The direction vector is the reusable artifact — it is what a future steering change would add to an activation. Reporting it now makes the observational and causal tabs talk to each other: the probe says "direction at layer 14 is w," and the patching tab can confirm "patching layer 14 shifts the output."

### 7. Where the probe lives in the code

**Decision:** A new `microscope/probing.py` module containing:
- `generate_concept_labels(...)` → `(prompts, labels)`
- `generate_quick_labels(...)` → `(prompts, labels)`
- `fit_probes(runtime, prompts, labels, C)` → `ProbeResult`
- `ProbeResult` dataclass: `per_layer_accuracy: pd.DataFrame`, `direction: dict[int, np.ndarray]` (layer → weight vector), `n_prompts: int`

The `app.py` Linear Probe tab calls these. The `analyse_batch()` helper lives in `analysis.py`.

## Risks / Trade-offs

- **[Small N in concept mode leads to overfitting]** → The L2 penalty slider lets the user increase regularization. The UI shows the number of training samples so the user can judge. Default `C=1.0` is a reasonable starting point.
- **[Batched forward memory]** → For N=50 prompts, seq=128, hidden=2048, float32: the hidden states tensor is `(50, 128, 2048)` ≈ 50 MB per layer, × 29 layers ≈ 1.4 GB on CPU. This is fine. The model weights (~3.5 GB for Qwen3-1.7B in bf16) are the dominant consumer.
- **[`analyse_batch` diverges from `analyse`]** → Two functions with similar names in the same module. Mitigation: document the difference clearly in the docstring. `analyse()` is for single-prompt interactive views; `analyse_batch()` is for batched training data. The hidden-states extraction logic is shared.
- **[Probe on final-token only]** → The probe targets the final-token hidden state, consistent with the rest of the workbench (logit lens, patching, attribution all target the final token). Probing other positions is a future extension, not a gap in this design.

## Migration Plan

None — new capability. No changes to existing `analyse()`, `Runtime`, or any existing tab. The Linear Probe tab is additive.

## Resolved Decisions

- **Probe reports in-sample accuracy only.** The classifier is fit and evaluated on the same data (no train/val split). This is consistent with the "quick experiment" ethos and keeps the implementation simple. For a small N this overestimates generalization, but the user judges significance from the accuracy curve shape, not from a held-out metric.
- **ROC AUC is not reported** in this change. Accuracy is the standard TDD metric. AUC is a follow-on if the user wants it.

## 1. Batched analysis helper

- [x] 1.1 Add `analyse_batch(runtime, prompts: list[str], max_length=128) -> BatchResult` in `microscope/analysis.py` that tokenizes a list of prompts, runs a single batched forward with `output_hidden_states=True`, and returns `(input_ids, hidden_states, logits)` without the single-prompt `labels` field. Verify: call on Qwen3-1.7B with 10 test prompts, confirm `hidden_states` is a tuple of 29 tensors each of shape `(10, seq, 2048)`.
- [x] 1.2 Confirm `analyse_batch` does not modify `analyse()`. Verify: run the existing single-prompt analysis in the app, confirm it still works and produces the same `AnalysisResult` as before.

## 2. Label generation: concept mode

- [x] 2.1 Implement `generate_concept_labels(pos_template: str, neg_template: str, fill_values: str) -> tuple[list[str], list[int]]` in `microscope/probing.py`. The `fill_values` is a comma-separated string; each fill value replaces `{X}` in both templates. Returns N positive prompts (label 1) and N negative prompts (label 0). Verify: call with `The capital of {X} is`, `The largest city in {X} is`, `France, Germany, Japan`, confirm it returns 6 prompts (3 pos + 3 neg) with correct labels.
- [x] 2.2 Confirm the generated prompts are balanced (equal number of pos and neg). Verify: for N fill values, the returned labels list has exactly N ones and N zeros.

## 3. Label generation: quick mode

- [x] 3.1 Implement `generate_quick_labels(prompts: list[str], substring: str) -> list[int]` in `microscope/probing.py`. Returns a label for each prompt: 1 if it contains the substring, 0 otherwise. Verify: call with prompts `["The capital of France is", "The capital of Germany is"]` and substring `France`, confirm it returns `[1, 0]`.

## 4. Shared fitting core

- [x] 4.1 Implement `fit_probes(runtime, prompts: list[str], labels: list[int], C: float = 1.0) -> ProbeResult` in `microscope/probing.py`. This function: (a) calls `analyse_batch` to get hidden states, (b) extracts the final-token hidden state per prompt per layer, (c) fits `LogisticRegression(C=C)` per layer, (d) returns a `ProbeResult` with `per_layer_accuracy: pd.DataFrame` (columns: `layer`, `accuracy`) and `direction: dict[int, np.ndarray]` (layer → weight vector). Verify: call on Qwen3-1.7B with 20 labeled prompts (10 pos + 10 neg from concept mode), confirm `per_layer_accuracy` has 28 rows and `direction` has 28 entries.
- [x] 4.2 Confirm the fitting uses in-sample evaluation (fit and evaluate on the same data, no train/val split). Verify: inspect the code — the classifier's `score()` is called on the same data it was fit on.
- [x] 4.3 Confirm both label modes (concept and quick) feed the same `fit_probes` function. Verify: call `fit_probes` with labels from `generate_concept_labels` and with labels from `generate_quick_labels`, confirm both work and return the same `ProbeResult` structure.

## 5. ProbeResult dataclass

- [x] 5.1 Define `ProbeResult` dataclass in `microscope/probing.py` with fields: `per_layer_accuracy: pd.DataFrame`, `direction: dict[int, np.ndarray]`, `n_prompts: int`, `n_positive: int`, `n_negative: int`. Verify: instantiate with sample data, confirm all fields are accessible.

## 6. UI integration

- [x] 6.1 Add a Linear Probe tab to `app.py` with: a mode selector (concept / quick), concept-mode inputs (positive template, negative template, fill values), quick-mode inputs (substring + prompts or template+fill values), a penalty slider (C ∈ [0.01, 100], log scale, default 1.0), and a "Fit probe" button. Verify: the Linear Probe tab shows all controls.
- [x] 6.2 Add the accuracy curve plot (plotly line chart, x=layer, y=accuracy) and a direction display (top-10 largest-magnitude components of the weight vector at the best-accuracy layer, with their indices) to the Linear Probe tab. Verify: after fitting, the tab shows the accuracy curve and the direction table.
- [x] 6.3 Add a "copy direction" button that copies the full weight vector (as a numpy array repr) to the clipboard for export. Verify: clicking the button copies the vector to the clipboard.

## 7. Verification

- [x] 7.1 Run the probe in concept mode on Qwen3-1.7B with `The capital of {X} is` / `The largest city in {X} is` and fill values `France, Germany, Japan, Spain, Italy, Brazil`. Confirm the accuracy curve shows a sensible shape (low at early layers, rising to high accuracy at mid-to-late layers). Verify: visually inspect the curve; accuracy should rise from near 0.5 (chance) at early layers to > 0.8 at some mid/late layer.
- [x] 7.2 Run the probe in quick mode with substring `France` on a set of prompts that do and don't contain "France". Confirm the accuracy curve shows where the surface form becomes decodable. Verify: visually inspect the curve; it should rise at an earlier or different layer than the concept mode curve (surface form is encoded earlier than the concept).
- [x] 7.3 Confirm the probe does not break any existing tabs. Verify: run the existing analysis (logit lens, attention, patching, etc.) after adding the probe, confirm all tabs still work.
- [x] 7.4 Confirm no new dependencies were added. Verify: `pyproject.toml` is unchanged; `scikit-learn` was already a dependency.

## 1. Submodule introspection

- [x] 1.1 Add `layer_modules(layer) -> tuple[Module|None, Module|None]` helper in `microscope/runtime.py` (or new `microscope/introspection.py`) that returns `(attn_module, mlp_module)` for the `self_attn`/`mlp` convention, or `(None, None)` if the convention is not found. Verify: import and call on a Qwen3-1.7B layer, confirm it returns the attention and MLP modules; call on a GPT-2 layer, confirm it returns `(None, None)`.

## 2. Target enum and injection machinery

- [x] 2.1 Define a `PatchTarget` enum (`RESIDUAL`, `ATTN_OUTPUT`, `MLP_OUTPUT`) in `microscope/interventions.py`. Verify: `PatchTarget.RESIDUAL`, `PatchTarget.ATTN_OUTPUT`, `PatchTarget.MLP_OUTPUT` are importable.
- [x] 2.2 Refactor the existing `patch_final_residual` injection logic into a generic `_inject_module_output(module, replacement, target)` helper that replaces the module's output (or final-token slice) with the captured value, supporting all three targets. Verify: existing single-layer residual patching still produces the same result as before (run the existing patching button in the app, confirm output matches).

## 3. Source capture via output_hidden_states

- [x] 3.1 Add `capture_source_hidden_states(runtime, source_prompt, max_length) -> dict[int, Tensor]` in `microscope/interventions.py` that runs one source forward with `output_hidden_states=True` and returns `{layer_index: final_token_state}` for every layer. Verify: call on Qwen3-1.7B with a test prompt, confirm the dict has 28 entries (one per layer) and each value has shape `(2048,)`.

## 4. Deduplicated patched-forward loop

- [x] 4.1 Implement `_patched_forwards(runtime, target_batch, source_states, target, max_length) -> list[Tensor]` that, for each layer, registers an injection hook on the target module, runs one target forward, reads `logits[0, -1]`, removes the hook, and returns the list of patched logits. Verify: call on Qwen3-1.7B with a source→target pair, confirm it returns 28 logit tensors.
- [x] 4.2 Confirm the dedup loop runs exactly 1 source forward + 1 baseline target forward + N patched forwards (N = number of layers). Verify: add a temporary forward-counter hook, run the score curve, confirm the count is 30 for a 28-layer model, then remove the counter.

## 5. Patching score curve function

- [x] 5.1 Implement `patching_score_curve(runtime, source_prompt, target_prompt, target=PatchTarget.RESIDUAL, max_length=128, k=10) -> pd.DataFrame` that orchestrates: source capture → baseline target forward → patched forwards → assembles a `DataFrame` with columns `layer`, `baseline_top1_token`, `patched_top1_token`, `baseline_top1_prob`, `patched_top1_prob`, `delta_logit`, `delta_prob`. Verify: call on Qwen3-1.7B with `The capital of France is` → `The capital of Germany is`, confirm the DataFrame has 28 rows and the columns are as specified.
- [x] 5.2 Confirm the curve reports both logit delta and probability delta. Verify: inspect the returned DataFrame for the `delta_logit` and `delta_prob` columns; confirm `delta_logit = patched_logit - baseline_logit` and `delta_prob = patched_prob - baseline_prob`.

## 6. GQA-aware attention-output patching

- [x] 6.1 Implement attention-output patching that hooks the attention module's output at the query-head level (16 heads for Qwen3-1.7B), not the KV-head level. The hook point is the attention output tensor (before `o_proj` collapse). Verify: call `patching_score_curve` with `target=PatchTarget.ATTN_OUTPUT` on Qwen3-1.7B, confirm it runs without error and returns 28 rows.
- [x] 6.2 Add a per-query-head selector to the UI (slider or dropdown for head 0–15) when the attention-output target is selected. Verify: in the app, selecting the attention-output target shows a head selector with 16 options.

## 7. Mean ablation

- [x] 7.1 Implement `mean_ablation(runtime, source_prompt, target_prompt, layer, target, mode, max_length) -> tuple[pd.DataFrame, pd.DataFrame]` where `mode` is `ZERO` or `MEAN`. For `ZERO`, set the target's final-token value to 0 during the target forward. For `MEAN`, set it to the mean of the source-run value. Verify: call on Qwen3-1.7B, confirm both modes run and return baseline + ablated top-token DataFrames.
- [x] 7.2 Document in the UI that for single-prompt runs, mean ablation is equivalent to patching (the source value is a single vector). Verify: the Patching tab shows a caption explaining this when mean ablation is selected.

## 8. UI integration

- [x] 8.1 Extend the Patching tab in `app.py` with: a target selector (residual / attention-output / MLP-output), a mean-ablation toggle (off / zero / mean), and a "Run score curve" button. Verify: the Patching tab shows all three controls.
- [x] 8.2 Add the score curve plot (plotly line chart, x=layer, y=delta_logit or delta_prob with a toggle) and the per-layer top-token table to the Patching tab. Verify: after running the score curve, the tab shows the curve plot and the table.
- [x] 8.3 Add graceful fallback: when `layer_modules` returns `(None, None)` for the loaded model, disable the attention-output and MLP-output selectors and show a caption explaining that module-level patching is unavailable for this architecture. Verify: load a GPT-2 model in the app, confirm only the residual target is selectable and the caption is shown.

## 9. Verification

- [x] 9.1 Run the existing patching button (single-layer residual) in the app and confirm the result is unchanged from before this change. Verify: the output matches the pre-change behaviour (same top-token table for the same layer/source/target).
- [x] 9.2 Run the score curve on Qwen3-1.7B with a known source→target pair (e.g. France→Germany) and confirm the curve shape is sensible (the peak should be in the mid-to-late layers, where the prediction is committed). Verify: visually inspect the curve; the delta should be near zero at early layers and peak at a specific mid/late layer.
- [x] 9.3 Run mean ablation (zero mode) on Qwen3-1.7B at a layer known to be important (from the score curve) and confirm the ablated result differs from the baseline. Verify: the ablated top-token table shows a different top-1 token or significantly different probabilities compared to the baseline.

import math
import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from microscope.analysis import (
    AnalysisResult,
    attention_head_similarity,
    attention_head_summary,
    attention_rollout,
    layer_prediction_metrics,
    pairwise_cosine_distances,
    representation_metrics,
)
from microscope.interventions import PatchTarget, ablation_effect_curve, patching_effect_matrix
from microscope.toolbox import probe_tool, tool_status_rows
from microscope.probing import fit_probes
from microscope.validity import calibration_metrics, compare_variants, prediction_uncertainty
from microscope.tuned_lens import decode_artifact, load_artifact
from microscope.components import component_cache_key, component_capabilities
from microscope.sae import ablate_feature, encode as encode_sae, load_artifact as load_sae_artifact
from microscope.adapters import bertviz_attention, circuitsvis_attention, provider_status


def fake_runtime() -> SimpleNamespace:
    head = torch.nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    return SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        final_norm=torch.nn.Identity(),
        lm_head=head,
    )


def fake_result(runtime: SimpleNamespace) -> AnalysisResult:
    hidden = (
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
    )
    logits = runtime.lm_head(hidden[-1]).detach()
    return AnalysisResult(
        input_ids=torch.tensor([[1, 2]]),
        labels=["0: a", "1: b"],
        logits=logits,
        hidden_states=hidden,
        attentions=(),
    )


class VisualizationHelperTests(unittest.TestCase):
    def test_layer_prediction_metrics_match_final_logits(self):
        runtime = fake_runtime()
        result = fake_result(runtime)

        metrics = layer_prediction_metrics(runtime, result, token_id=1)
        final = result.logits[0].softmax(dim=-1)[:, 1]

        self.assertTrue(
            all(
                math.isclose(actual, expected, rel_tol=1e-6)
                for actual, expected in zip(
                    metrics[metrics["layer"] == 0]["probability"], final.tolist()
                )
            )
        )


    def test_representation_metrics_and_pairwise_distances_have_token_shapes(self):
        runtime = fake_runtime()
        result = fake_result(runtime)

        norms = representation_metrics(result, "norm")
        changes = representation_metrics(result, "cosine_change")
        distances = pairwise_cosine_distances(result, layer=0)

        self.assertEqual(norms.shape[0], 2)
        self.assertEqual(changes.shape[0], 2)
        self.assertEqual(distances.shape, (2, 2))
        self.assertEqual(list(distances.index), result.labels)


    def test_patching_effect_matrix_preserves_missing_targets(self):
        curve = pd.DataFrame(
            {"layer": [0, 1], "delta_logit": [0.1, -0.2], "delta_prob": [0.01, -0.02]}
        )
        matrix = patching_effect_matrix(
            [{"curve": curve, "patch_target_display": "Residual"}],
            "delta_logit",
        )

        self.assertEqual(matrix.loc["Residual", 0], 0.1)
        self.assertNotIn("Attention output", matrix.index)

    def test_attention_summaries_rollout_and_similarity(self):
        attentions = (
            torch.tensor([[[[1.0, 0.0], [0.25, 0.75]], [[0.5, 0.5], [0.0, 1.0]]]]),
            torch.tensor([[[[0.75, 0.25], [0.5, 0.5]], [[1.0, 0.0], [0.2, 0.8]]]]),
        )

        summary = attention_head_summary(attentions, ["0: a", "1: b"])
        rollout = attention_rollout(attentions)
        similarity = attention_head_similarity(attentions, layer=0)

        self.assertEqual(len(summary), 8)
        self.assertTrue((summary["entropy"] >= 0).all())
        self.assertEqual(rollout.shape, (2, 2))
        self.assertTrue(torch.allclose(rollout.sum(dim=-1), torch.ones(2)))
        self.assertEqual(similarity.shape, (2, 2))

    def test_tool_status_includes_capability_and_compatibility_fields(self):
        rows = tool_status_rows({"TransformerLens": "checked"})

        transformer_lens = next(row for row in rows if row["tool"] == "TransformerLens")
        self.assertIn("capability", transformer_lens)
        self.assertEqual(transformer_lens["compatibility"], "checked")
        self.assertTrue(any("evaluation candidate" in row["tool"] for row in rows))

    def test_visualisation_candidates_report_installation_and_adapter_state(self):
        rows = tool_status_rows()
        candidates = {
            row["tool"]: row
            for row in rows
            if "evaluation candidate" in row["tool"]
        }

        self.assertEqual(
            set(candidates),
            {
                "SAELens (evaluation candidate)",
                "Pyvene (evaluation candidate)",
                "BertViz (evaluation candidate)",
                "CircuitsVis (evaluation candidate)",
                "Tuned Lens (evaluation candidate)",
            },
        )
        for row in candidates.values():
            self.assertIn("package", row)
            self.assertIn("installed", row)
            self.assertTrue(row["adapter_enabled"])
            self.assertEqual(row["integration_status"], "Integrated")
            self.assertTrue(row["visualization_scope"])
            self.assertFalse(row["compatibility_checked"])

    def test_unintegrated_tool_status_does_not_require_package_import(self):
        tool = "BertViz (evaluation candidate)"
        absent = {row["tool"]: False for row in tool_status_rows()}
        with patch("microscope.toolbox.installed_tools", return_value=absent):
            message = probe_tool(tool, "test-model")

        self.assertIn("not installed", message)
        self.assertIn("Native views remain available", message)

    def test_installed_candidate_is_still_reported_as_unintegrated(self):
        tool = "CircuitsVis (evaluation candidate)"
        installed = {row["tool"]: False for row in tool_status_rows()}
        installed[tool] = True
        with patch("microscope.toolbox.installed_tools", return_value=installed):
            message = probe_tool(tool, "test-model")

        self.assertIn("ready for an explicit compatibility check", message)

    def test_optional_probe_failure_preserves_core_visualisation_message(self):
        with patch("microscope.toolbox._probe_tool", side_effect=RuntimeError("provider failed")):
            message = probe_tool("BertViz (evaluation candidate)", "test-model")

        self.assertIn("provider failed", message)
        self.assertIn("core visualisations remain available", message)

    def test_integrated_gradients_missing_extra_is_actionable(self):
        from microscope.interventions import integrated_gradients

        if importlib.util.find_spec("captum") is not None:
            self.skipTest("Captum is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "captum"):
            integrated_gradients(None, "test")

    def test_uncertainty_and_calibration_metrics_are_explicit(self):
        metrics = prediction_uncertainty(torch.tensor([2.0, 1.0, 0.0]))
        self.assertEqual(metrics["top1_id"], 0)
        self.assertGreater(metrics["top1_margin"], 0)

        summary, bins = calibration_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        self.assertEqual(summary["n_examples"], 4)
        self.assertEqual(int(bins["count"].sum()), 4)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_variant_comparison_preserves_first_answer_reference(self):
        result = compare_variants([
            {"variant": "base", "token": " Paris", "probability": 0.8, "entropy": 1.0},
            {"variant": "masked", "token": " London", "probability": 0.4, "entropy": 1.5},
        ])
        self.assertTrue(bool(result.iloc[0]["agreement_with_first"]))
        self.assertFalse(bool(result.iloc[1]["agreement_with_first"]))

    def test_ablation_curve_cleans_hooks_and_keeps_baseline_metadata(self):
        class ToyModel(torch.nn.Module):
            def __init__(self, layers, head):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)
                self.head = head

            def forward(self, **kwargs):
                state = kwargs["input_ids"].float()
                for layer in self.layers:
                    state = layer(state)
                return SimpleNamespace(logits=self.head(state))

        layers = [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        head = torch.nn.Linear(2, 3, bias=False)
        runtime = SimpleNamespace(
            model=ToyModel(layers, head), layers=layers,
            device=torch.device("cpu"),
        )
        with patch("microscope.interventions.tokenize", return_value={"input_ids": torch.ones(1, 1, 2)}):
            curve = ablation_effect_curve(runtime, "test", PatchTarget.RESIDUAL)
        self.assertEqual(len(curve), 2)
        self.assertIn("baseline_logit", curve)
        self.assertTrue(all(not layer._forward_hooks for layer in layers))

    def test_attention_head_ablation_uses_projection_pre_hook(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Linear(4, 4, bias=False)

            def forward(self, value):
                return self.o_proj(value)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

            def forward(self, value):
                return self.self_attn(value)

        class Model(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)
                self.config = SimpleNamespace(num_attention_heads=2)
                self.head = torch.nn.Linear(4, 3, bias=False)

            def forward(self, **kwargs):
                value = kwargs["input_ids"].float()
                for layer in self.layers:
                    value = layer(value)
                return SimpleNamespace(logits=self.head(value))

        layers = [Layer()]
        runtime = SimpleNamespace(model=Model(layers), layers=layers, device=torch.device("cpu"))
        with patch("microscope.interventions.tokenize", return_value={"input_ids": torch.ones(1, 1, 4)}):
            curve = ablation_effect_curve(runtime, "test", PatchTarget.ATTN_OUTPUT, head_idx=1)
        self.assertEqual(curve.iloc[0]["head_idx"], 1)

    def test_native_tuned_lens_artifact_requires_exact_model_and_layers(self):
        payload = {
            "model_name": "test-model",
            "hidden_size": 3,
            "weights": [torch.eye(3)],
            "biases": [torch.zeros(3)],
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as file:
            torch.save(payload, file.name)
            artifact = load_artifact(file.name, "test-model", 3)
        runtime = fake_runtime()
        translated = decode_artifact(runtime, fake_result(runtime).hidden_states, artifact, 1)
        self.assertEqual(len(translated), 1)
        self.assertEqual(int(translated.iloc[0]["token_id"]), 1)

    def test_component_capabilities_and_cache_identity_are_explicit(self):
        runtime = SimpleNamespace(layers=[object()], model=SimpleNamespace(config=SimpleNamespace()))
        capabilities = component_capabilities(runtime, 0)
        self.assertFalse(capabilities.mlp_neurons)
        self.assertIn("not expose", capabilities.reason)
        key = component_cache_key("model", "prompt", 2, "mlp-neuron", (1,), neuron=4, intervention="zero")
        self.assertIn("mlp-neuron", key)

    def test_sae_artifact_validation_and_feature_ablation(self):
        payload = {
            "model_name": "test-model", "layer": 1, "hidden_size": 3,
            "encoder": torch.eye(3), "decoder": torch.eye(3), "threshold": 0.0,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as file:
            torch.save(payload, file.name)
            artifact = load_sae_artifact(file.name, "test-model", 1, 3)
        activations = torch.tensor([[1.0, 0.0, 2.0]])
        features = encode_sae(activations, artifact)
        ablated = ablate_feature(activations, artifact, 2)
        self.assertEqual(features.shape, (1, 3))
        self.assertEqual(float(ablated[0, 2]), 0.0)

    def test_optional_visual_adapters_return_labeled_html(self):
        attention = (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),)
        bert = bertviz_attention(attention, ["0: a", "1: b"])
        circuits = circuitsvis_attention(attention[0], ["0: a", "1: b"])
        self.assertEqual(bert.kind, "html")
        self.assertEqual(circuits.kind, "html")
        self.assertIn("tokens", bert.labels)
        self.assertTrue(provider_status("BertViz").adapter_ready)

    def test_probe_reports_heldout_and_training_only_modes(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, prompts, **kwargs):
                count = len(prompts)
                return Batch(
                    input_ids=torch.ones(count, 2, dtype=torch.long),
                    attention_mask=torch.ones(count, 2, dtype=torch.long),
                )

        class Model:
            def __call__(self, **kwargs):
                count = kwargs["input_ids"].shape[0]
                values = torch.arange(count, dtype=torch.float32).view(count, 1, 1)
                hidden = (
                    torch.zeros(count, 2, 2),
                    values.expand(count, 2, 2),
                    values.expand(count, 2, 2),
                )
                return SimpleNamespace(hidden_states=hidden, logits=torch.zeros(count, 2, 3))

        runtime = SimpleNamespace(
            tokenizer=Tokenizer(),
            model=Model(),
            device=torch.device("cpu"),
            layers=[object(), object()],
        )
        heldout = fit_probes(runtime, [str(i) for i in range(8)], [0, 0, 0, 0, 1, 1, 1, 1])
        training_only = fit_probes(runtime, ["a", "b"], [0, 1])

        self.assertTrue(heldout.heldout_supported)
        self.assertIn("heldout_accuracy", heldout.per_layer_accuracy)
        self.assertFalse(training_only.heldout_supported)
        self.assertEqual(training_only.evaluation, "training-only evaluation")

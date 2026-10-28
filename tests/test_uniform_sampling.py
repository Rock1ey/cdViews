"""CPU checks for sampling and inference wiring; no model weights needed."""

import ast
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cdviews.view_sampling import question_seed, uniform_views
import run_scanqa


REPO = Path(__file__).resolve().parents[1]


class SamplingTests(unittest.TestCase):
    def test_ignores_ranking_without_replacement(self):
        frames = [f"{i}.jpg" for i in range(30)]
        selected = uniform_views(frames, 9, 42, "val", "q1")
        self.assertEqual(selected, uniform_views(frames[::-1], 9, 42, "val", "q1"))
        self.assertEqual(len(set(selected)), 9)
        self.assertTrue(set(selected) <= set(frames))
        self.assertNotEqual(selected, uniform_views(frames, 9, 43, "val", "q1"))

    def test_smoke_full_and_shards_select_identical_views(self):
        frames = [f"{i}.jpg" for i in range(30)]
        questions = [f"q{i}" for i in range(10)]
        full = {q: uniform_views(frames, 9, 42, "val", q) for q in questions}
        for q in questions[::-1] + questions[::2] + questions[:2]:
            self.assertEqual(full[q], uniform_views(frames, 9, 42, "val", q))
        self.assertNotEqual(question_seed(42, "val", "q1", "views"),
                            question_seed(42, "val", "q1", "decode"))

    def test_small_candidate_pool(self):
        selected = uniform_views(["a.jpg", "b.jpg"], 9, 42, "val", "q1")
        self.assertEqual(set(selected), {"a.jpg", "b.jpg"})

    def test_uniform_inference_skips_selector_and_nms(self):
        # Execute the actual inference function with a fake LVLM. Extracting the
        # function avoids importing CUDA/transformers packages on the Mac.
        source = (REPO / "cdviews/qa_inference.py").read_text().replace(
            "test_mode = ['test_w_obj', 'test_wo_obj'] if args.dataset == 'ScanQA' else ['test', ]",
            "test_mode = ['val']")
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "eval_model")
        model, tokenizer, processor = MagicMock(), MagicMock(), MagicMock()
        tokenizer.batch_decode.return_value = [" Chair\nextra words"]
        forbidden = MagicMock(side_effect=AssertionError("Uniform invoked selector/features/poses/NMS"))
        namespace = dict(os=os, json=json, torch=SimpleNamespace(inference_mode=nullcontext),
                         transformers=MagicMock(), disable_torch_init=lambda: None,
                         get_model_name_from_path=lambda p: p,
                         load_pretrained_model=lambda *a: (tokenizer, model, processor, 2048),
                         uniform_views=uniform_views, question_seed=question_seed,
                         ViewSelector=forbidden, ViewLabelDataset=forbidden,
                         DataLoader=forbidden, calculate_view_distance=forbidden, viewNMS=forbidden,
                         pd=SimpleNamespace(read_csv=forbidden), Image=MagicMock(),
                         preprocess_qwen=MagicMock(), tqdm=lambda rows: rows,
                         get_scanqa=lambda *a, **kw: [dict(scene_id="scene", question_id="q1", question="What?")])
        exec(compile(ast.Module(body=[function], type_ignores=[]), "qa_inference.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            ranking = Path(tmp) / "ranking.json"
            ranking.write_text(json.dumps({"q1": [f"{i}.jpg" for i in range(20)]}))
            args = SimpleNamespace(LVLM_ckpt="fake", model_base=None, dataset="ScanQA",
                                   ranked_view_file=str(ranking), answers_file=str(Path(tmp) / "answers.json"),
                                   view_selection="uniform", input_views=9, seed=42,
                                   image_folder=tmp, temperature=0, top_p=None, num_beams=1)
            with redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)
            answers = json.loads(Path(args.answers_file).read_text())
            trace = json.loads(Path(args.answers_file + ".views.json").read_text())
            self.assertEqual(answers[0]["answer_top10"][0], "chair")
            self.assertEqual(len(trace["questions"][0]["images"]), 9)
            self.assertFalse(model.generate.call_args.kwargs["do_sample"])
            forbidden.assert_not_called()

            # Retrieval must read its own BLIP order and bypass NMS as well.
            blip_ranking = Path(tmp) / "blip.json"
            blip_ranking.write_text(json.dumps({"q1": ["8.jpg", "3.jpg", "5.jpg"]}))
            args.view_selection = "retrieval"
            args.retrieval_ranked_view_file = str(blip_ranking)
            args.answers_file = str(Path(tmp) / "retrieval_answers.json")
            args.input_views = 2
            with redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)
            trace = json.loads(Path(args.answers_file + ".views.json").read_text())
            self.assertEqual(trace["questions"][0]["images"], ["8.jpg", "3.jpg"])
            forbidden.assert_not_called()

            # Soft selection is consumed unchanged, without a second NMS. All
            # images must enter one generation call, not one call per view.
            soft_file = Path(tmp) / "soft.json"
            soft_file.write_text(json.dumps({"q1": ["0.jpg", "9.jpg", "4.jpg"]}))
            args.view_selection = "soft"
            args.soft_selected_view_file = str(soft_file)
            args.answers_file = str(Path(tmp) / "soft_answers.json")
            args.input_views = 3
            model.generate.reset_mock()
            with redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)
            trace = json.loads(Path(args.answers_file + ".views.json").read_text())
            self.assertEqual(trace["questions"][0]["images"], ["0.jpg", "9.jpg", "4.jpg"])
            model.generate.assert_called_once()
            self.assertEqual(len(model.generate.call_args.kwargs["images"]), 3)
            self.assertEqual(trace["questions"][0]["candidate_count"], 20)
            forbidden.assert_not_called()
            args.view_selection = "coverage"
            model.generate.reset_mock()
            with redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)
            trace = json.loads(Path(args.answers_file + ".views.json").read_text())
            self.assertEqual(trace["method"], "coverage")
            self.assertEqual(trace["questions"][0]["images"], ["0.jpg", "9.jpg", "4.jpg"])
            model.generate.assert_called_once()
            self.assertEqual(len(model.generate.call_args.kwargs["images"]), 3)
            forbidden.assert_not_called()
            soft_file.write_text(json.dumps({"q1": ["0.jpg", "0.jpg", "4.jpg"]}))
            with self.assertRaisesRegex(ValueError, "Invalid soft selection"), redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)

            # Cached cdViews still uses geometric NMS, without loading selector
            # weights or features merely to consume the same cached ranking.
            geometry = MagicMock()
            nms = MagicMock(return_value=["1.jpg", "2.jpg"])
            namespace.update(calculate_view_distance=geometry, viewNMS=nms)
            args.view_selection = "cdviews"
            args.view_distance_folder = str(Path(tmp) / "distance")
            args.answers_file = str(Path(tmp) / "cdviews_answers.json")
            with redirect_stdout(io.StringIO()):
                namespace["eval_model"](args)
            geometry.assert_called_once()
            nms.assert_called_once()
            trace = json.loads(Path(args.answers_file + ".views.json").read_text())
            self.assertEqual(trace["questions"][0]["images"], ["1.jpg", "2.jpg"])
            forbidden.assert_not_called()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.annotation = self.root / "ScanQA/ScanQA_v1.0/ScanQA_v1.0_val.json"
        self.annotation.parent.mkdir(parents=True)
        self.rows = [dict(question_id=f"q{i}", scene_id="scene", question="What?", answers=["chair"])
                     for i in range(4675)]
        self.annotation.write_text(json.dumps(self.rows))
        colors = self.root / "scannetv2/frames_square/scene/color"
        colors.mkdir(parents=True)
        for i in range(20):
            (colors / f"{i}.jpg").touch()
        self.ranking = self.root / "ScanQA_val_view_ranking.json"
        self.ranking.write_text(json.dumps({r["question_id"]: [f"{i}.jpg" for i in range(20)]
                                            for r in self.rows}))
        self.phases = []

    def launch(self, method="uniform", stage="smoke"):
        argv = ["run_scanqa.py", "--repo", str(REPO), "--data-root", str(self.root),
                "--run-dir", str(self.root / method), "--stage", stage,
                "--view-selection", method, "--temperature", "0", "--seed", "42"]

        def fake_process(command, **kwargs):
            cfg = run_scanqa.yaml.safe_load(Path(command[-1]).read_text())
            self.phases.append(Path(command[2]).name)
            rows = json.loads(Path(cfg["ScanQA_file"]["val"]).read_text())
            if Path(command[2]).name == "blip_retrieval.py":
                candidates = run_scanqa.read_json(Path(cfg["ranked_view_file"].format("ScanQA", "val")))
                run_scanqa.write_json(Path(cfg["retrieval_ranked_view_file"].format("ScanQA", "val")),
                                     {qid: frames[::-1] for qid, frames in candidates.items()})
                return SimpleNamespace(pid=1, poll=lambda: 0, wait=lambda timeout: 0, returncode=0)
            if Path(command[2]).name == "soft_view_preparation.py":
                candidates = run_scanqa.read_json(Path(cfg["ranked_view_file"].format("ScanQA", "val")))
                run_scanqa.write_json(Path(cfg["soft_selected_view_file"].format("ScanQA", "val")),
                                     {qid: frames[:cfg["input_views"]] for qid, frames in candidates.items()})
                return SimpleNamespace(pid=1, poll=lambda: 0, wait=lambda timeout: 0, returncode=0)
            if method == "retrieval":
                self.assertTrue(Path(cfg["retrieval_ranked_view_file"].format("ScanQA", "val")).is_file())
            if method in ("soft", "coverage"):
                self.assertTrue(Path(cfg["soft_selected_view_file"].format("ScanQA", "val")).is_file())
            output = Path(cfg["answers_file"].format("ScanQA", "val"))
            run_scanqa.write_json(output, [dict(scene_id=r["scene_id"], question_id=r["question_id"],
                                                answer_top10=["chair"] * 10) for r in rows])
            run_scanqa.write_json(Path(str(output) + ".views.json"), dict(
                method=method, seed=42, input_views=cfg["input_views"], split="val",
                questions=[dict(question_id=r["question_id"], scene_id=r["scene_id"],
                                candidate_count=20, images=["0.jpg"]) for r in rows]))
            return SimpleNamespace(pid=1, poll=lambda: 0, wait=lambda timeout: 0, returncode=0)

        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"CDV_GPU": "3"}), \
             patch.object(run_scanqa.subprocess, "run", return_value=SimpleNamespace(stdout="test\n")), \
             patch.object(run_scanqa.subprocess, "Popen", side_effect=fake_process), \
             redirect_stdout(io.StringIO()):
            run_scanqa.main()
        return self.root / method / stage

    def test_launcher_snapshots_inputs_and_merges_complete_results(self):
        original = self.ranking.read_bytes()
        stage = self.launch(stage="full")
        cfg = run_scanqa.yaml.safe_load((stage / "gpu3/QA.yaml").read_text())
        self.assertEqual((cfg["view_selection"], cfg["input_views"], cfg["seed"], cfg["temperature"]),
                         ("uniform", 9, 42, 0))
        self.assertTrue(Path(cfg["ranked_view_file"].format("ScanQA", "val")).is_relative_to(stage.resolve()))
        self.assertEqual(len(run_scanqa.read_json(stage / "ScanQA_val_answers.json")), 4675)
        self.assertEqual(len(run_scanqa.read_json(stage / "ScanQA_val_answers.json.views.json")["questions"]), 4675)
        self.assertEqual(self.ranking.read_bytes(), original)
        self.assertEqual(run_scanqa.read_json(stage / "experiment.json")["gpus"], ["3"])
        with self.assertRaises(FileExistsError):
            self.launch(stage="full")

    def test_cdviews_config_still_available(self):
        stage = self.launch(method="cdviews")
        self.assertEqual(run_scanqa.read_json(stage / "experiment.json")["method"], "cdviews")
        self.assertEqual(len(run_scanqa.read_json(stage / "ScanQA_val_answers.json")), 2)

    def test_retrieval_runs_before_qa_and_defaults_to_17(self):
        model_dir = self.root / "model/blip-itm-large-coco"
        model_dir.mkdir(parents=True)
        for name in ("pytorch_model.bin", "config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
            (model_dir / name).touch()
        stage = self.launch(method="retrieval")
        self.assertEqual(self.phases, ["blip_retrieval.py", "qa_inference_selected_splits.py"])
        cfg = run_scanqa.yaml.safe_load((stage / "gpu3/QA.yaml").read_text())
        self.assertEqual(cfg["input_views"], 17)
        self.assertEqual(cfg["inference_splits"], ["val"])

    def test_retrieval_requires_weights_before_creating_run(self):
        with self.assertRaisesRegex(FileNotFoundError, "download_blip"):
            self.launch(method="retrieval")
        self.assertFalse((self.root / "retrieval/smoke").exists())

    def test_soft_runs_before_qa_and_preserves_source_ranking(self):
        model_dir = self.root / "model/blip-itm-large-coco"
        model_dir.mkdir(parents=True)
        for name in ("pytorch_model.bin", "config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
            (model_dir / name).touch()
        original = self.ranking.read_bytes()
        stage = self.launch(method="soft")
        self.assertEqual(self.phases, ["soft_view_preparation.py", "qa_inference_selected_splits.py"])
        cfg = run_scanqa.yaml.safe_load((stage / "gpu3/QA.yaml").read_text())
        self.assertEqual((cfg["input_views"], cfg["soft_penalty"], cfg["soft_rank_scale"], cfg["soft_pose_scale"]),
                         (9, 0.5, 10, 0.5))
        self.assertEqual(self.ranking.read_bytes(), original)
        self.assertEqual((stage / "blip_retrieval.py").read_bytes(), (REPO / "cdviews/blip_retrieval.py").read_bytes())
        self.assertTrue((stage / "soft_selection.py").is_file())
        self.assertFalse(run_scanqa.read_json(stage / "experiment.json")["soft"]["hard_nms"])

    def test_coverage_runs_before_qa_with_separate_output_and_parameters(self):
        model_dir = self.root / "model/blip-itm-large-coco"
        model_dir.mkdir(parents=True)
        for name in ("pytorch_model.bin", "config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
            (model_dir / name).touch()
        stage = self.launch(method="coverage")
        self.assertEqual(self.phases, ["soft_view_preparation.py", "qa_inference_selected_splits.py"])
        cfg = run_scanqa.yaml.safe_load((stage / "gpu3/QA.yaml").read_text())
        self.assertEqual(cfg["view_selection"], "coverage")
        self.assertEqual(cfg["input_views"], 9)
        self.assertTrue(cfg["soft_selected_view_file"].endswith("coverage_views.json"))
        self.assertEqual((cfg["coverage_weight"], cfg["coverage_temperature"], cfg["coverage_text_weight"]),
                         (0.5, 0.07, 1.0))
        metadata = run_scanqa.read_json(stage / "experiment.json")
        self.assertEqual(metadata["coverage"]["weight"], 0.5)
        self.assertEqual(metadata["soft"]["penalty"], 0.5)

    def test_rejects_duplicate_candidates(self):
        ranking = run_scanqa.read_json(self.ranking)
        ranking["q0"].append("0.jpg")
        run_scanqa.write_json(self.ranking, ranking)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            self.launch()

    def test_rejects_incomplete_validation_annotations(self):
        self.annotation.write_text(json.dumps(self.rows[:2]))
        with self.assertRaisesRegex(ValueError, "4675"):
            self.launch()


if __name__ == "__main__":
    unittest.main()

"""Offline analysis must preserve the evaluator's strict matching and view order."""

import json
import ast
from pathlib import Path
import tempfile
import unittest

from analyze_scanqa import compare, draw_case, load_run, question_type, replay_view_nms, sha256


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gold = [{"question_id": str(i), "scene_id": "scene", "question": "What color is it?",
                      "answers": ["red", "scarlet"]} for i in range(4)]
        self.annotations = self.root / "val.json"
        self.annotations.write_text(json.dumps(self.gold))

    def tearDown(self):
        self.temp.cleanup()

    def run_file(self, name, answers, images=None):
        path = self.root / (name + ".json")
        path.write_text(json.dumps([{"question_id": str(i), "scene_id": "scene", "answer_top10": [a]}
                                    for i, a in reversed(list(enumerate(answers)))]))
        if images:
            Path(str(path) + ".views.json").write_text(json.dumps({
                "split": "val", "input_views": 2,
                "questions": [{"question_id": str(i), "scene_id": "scene", "candidate_count": 3,
                               "images": images} for i in range(4)]}))
        return path

    def test_transitions_and_strict_em(self):
        a, gold = load_run("a", self.run_file("a", ["Red", "red", "scarlet", "blue"]), self.annotations, 4)
        b, _ = load_run("b", self.run_file("b", ["red", "red ", "red", "green"]), self.annotations, 4)
        rows, summary = compare(gold, a, b)
        self.assertEqual([r["category"] for r in rows], ["fixed", "regressed", "both_correct", "both_wrong"])
        self.assertEqual(summary["delta_em_pp"], 0)
        self.assertEqual(summary["fixed"], 1)
        self.assertEqual(rows[0]["view_jaccard"], "")

    def test_views_preserve_order_and_set(self):
        a, gold = load_run("a", self.run_file("a", ["red"] * 4, ["1.jpg", "2.jpg"]), self.annotations, 4)
        b, _ = load_run("b", self.run_file("b", ["red"] * 4, ["2.jpg", "1.jpg"]), self.annotations, 4)
        rows, _ = compare(gold, a, b)
        self.assertEqual(rows[0]["view_jaccard"], 1)
        self.assertEqual(rows[0]["same_view_order"], 0)
        self.assertEqual(rows[0]["same_view_set"], 1)

    def test_reject_wrong_size_and_missing_prediction(self):
        path = self.run_file("a", ["red"] * 4)
        with self.assertRaisesRegex(ValueError, "4675"):
            load_run("a", path, self.annotations)
        path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
        with self.assertRaisesRegex(ValueError, "Question IDs"):
            load_run("a", path, self.annotations, 4)

    def test_reject_duplicate_view_ids(self):
        path = self.run_file("a", ["red"] * 4, ["1.jpg"])
        trace_path = Path(str(path) + ".views.json")
        trace = json.loads(trace_path.read_text())
        trace["questions"].append(trace["questions"][0])
        trace_path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_run("a", path, self.annotations, 4)

    def test_case_sheet_missing_image_is_explicit(self):
        from PIL import Image
        a, gold = load_run("a", self.run_file("a", ["red"] * 4, ["1.jpg"]), self.annotations, 4)
        b, _ = load_run("b", self.run_file("b", ["blue"] * 4, ["2.jpg"]), self.annotations, 4)
        rows, _ = compare(gold, a, b)
        folder = self.root / "scene" / "color"
        folder.mkdir(parents=True)
        Image.new("RGB", (100, 100), "red").save(folder / "1.jpg")
        font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
        missing = draw_case(rows[0], self.root, self.root / "case.png", font if font.exists() else None)
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].endswith("2.jpg"))
        with Image.open(self.root / "case.png") as image:
            self.assertEqual(image.width, 1080)

    def test_lexical_groups(self):
        self.assertEqual(question_type("How many chairs?"), "how many")
        self.assertEqual(question_type(" Where is the chair?"), "where")

    def reconstruction_inputs(self):
        import pandas as pd
        images = ["1.jpg", "2.jpg", "3.jpg"]
        ranking = self.root / "ScanQA_val_view_ranking.json"
        ranking.write_text(json.dumps({str(i): images for i in range(4)}))
        # The second view is suppressed; equality to 0.5 must NOT suppress the third.
        distances = pd.DataFrame([[0, .1, .5], [.1, 0, .9], [.5, .9, 0]], index=images, columns=images)
        distances.to_csv(self.root / "scene.csv")
        return ranking, distances

    def test_reconstruction_is_nms_not_topk(self):
        ranking, _ = self.reconstruction_inputs()
        path = self.run_file("a", ["red"] * 4)
        run, gold = load_run("a", path, self.annotations, 4, ranking_path=ranking,
                             distance_dir=self.root, cdviews_budget=9)
        self.assertEqual(run["views"]["0"]["images"], ["1.jpg", "3.jpg"])
        self.assertEqual(run["views"]["0"]["candidate_count"], 3)
        self.assertEqual(run["view_source"]["kind"], "reconstructed_cdviews_nms")
        self.assertFalse(Path(str(path) + ".views.json").exists())
        rows, _ = compare(gold, run, run)
        self.assertEqual(rows[0]["view_jaccard"], 1)

    def test_replay_matches_original_inference_function(self):
        _, distances = self.reconstruction_inputs()
        path = Path(__file__).resolve().parents[1] / "cdviews/qa_inference.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "viewNMS")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        for candidates in (["1.jpg", "2.jpg", "3.jpg"], ["3.jpg", "2.jpg", "1.jpg"]):
            for budget in (1, 2, 9):
                for threshold in (0, .5, 1):
                    self.assertEqual(replay_view_nms(candidates, distances, budget, threshold),
                                     namespace["viewNMS"](candidates, distances, budget, threshold))

    def test_recorded_views_take_precedence(self):
        path = self.run_file("a", ["red"] * 4, ["2.jpg", "1.jpg"])
        run, _ = load_run("a", path, self.annotations, 4, ranking_path=self.root / "nonexistent.json")
        self.assertEqual(run["view_source"]["kind"], "recorded")
        self.assertEqual(run["views"]["0"]["images"], ["2.jpg", "1.jpg"])

    def test_reconstruction_verifies_original_ranking(self):
        ranking, _ = self.reconstruction_inputs()
        path = self.run_file("a", ["red"] * 4)
        experiment = {"method": "cdviews", "input_views": 9,
                      "inputs": {"val_candidates": {"sha256": sha256(ranking)}}}
        (self.root / "experiment.json").write_text(json.dumps(experiment))
        run, _ = load_run("a", path, self.annotations, 4, ranking_path=ranking, distance_dir=self.root)
        self.assertTrue(run["view_source"]["ranking_hash_verified_against_experiment"])
        ranking.write_text(ranking.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_run("a", path, self.annotations, 4, ranking_path=ranking, distance_dir=self.root)

    def test_missing_distances_do_not_fall_back_to_topk(self):
        ranking, _ = self.reconstruction_inputs()
        path = self.run_file("a", ["red"] * 4)
        with self.assertRaisesRegex(FileNotFoundError, "view-distance CSV"):
            load_run("a", path, self.annotations, 4, ranking_path=ranking, distance_dir=self.root / "missing")


if __name__ == "__main__":
    unittest.main()

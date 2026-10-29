"""R2 model, data, real CPU training/resume and two-process DDP checks."""

import ast
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict
import unittest

import torch

from cdviews.r2_model import QueryViewSelector, binary_losses
from cdviews.r2_data import (collate_questions, model_inputs, scene_split, fingerprint,
                             write_json, read_json, average_precision)
from cdviews.prepare_r2_features import text_token_ids, build_plan, immutable_json
from cdviews.rank_r2 import merge_rankings


REPO = Path(__file__).resolve().parents[1]


def tiny_config():
    return dict(input_channel=8, embedding_dim=8, flat_out_size=16, num_heads=2, num_layers=1, num_queries=2)


def make_cache(root):
    torch.manual_seed(3)
    root.mkdir()
    spec = {"fixture": "tiny frozen features"}
    write_json(root / "config.json", spec)
    rows = []
    for scene_index in range(5):
        scene = f"scene{scene_index}"
        folder = root / "images" / scene
        folder.mkdir(parents=True)
        for i in range(4):
            torch.save(torch.randn(3 + i % 2, 8), folder / f"{i}.jpg.pt")
        for question_index in range(2):
            qid = f"q{scene_index}_{question_index}"
            text_folder = root / "questions/train"
            text_folder.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(4, 8), text_folder / f"{qid}.pt")
            rows.append(dict(question_id=qid, scene_id=scene, question="What?",
                             positive=["0.jpg", "1.jpg"], negative=["2.jpg", "3.jpg"]))
    write_json(root / "train.json", dict(split="train", cache_id=fingerprint(spec), input_channel=8, rows=rows))
    return rows


class R2ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_visual_attention_and_queries_receive_gradients(self):
        model = QueryViewSelector(**tiny_config())
        batch = collate_questions([([torch.randn(4, 8), torch.randn(3, 8)], torch.randn(5, 8), torch.tensor([1., 0.]))])
        binary_losses(model(**model_inputs(batch, "cpu")), batch["labels"]).mean().backward()
        for parameter in (model.queries, model.question_query.weight,
                          model.fusion_backbone.dec_list[0].mhatt1.linear_q.weight,
                          model.fusion_backbone.dec_list[0].mhatt1.linear_k.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_padding_does_not_change_scores(self):
        model = QueryViewSelector(**tiny_config()).eval()
        image, text = torch.randn(4, 8), torch.randn(3, 8)
        one = collate_questions([([image], text, torch.ones(1))])
        padded = collate_questions([([image], text, torch.ones(1)),
                                   ([torch.randn(8, 8)], torch.randn(9, 8), torch.zeros(1))])
        with torch.no_grad():
            first = model(**model_inputs(one, "cpu"))[0]
            second = model(**model_inputs(padded, "cpu"))[0]
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)

    def test_small_batch_can_overfit(self):
        model = QueryViewSelector(**tiny_config())
        batch = collate_questions([([torch.randn(4, 8) for _ in range(4)], torch.randn(4, 8), torch.tensor([1., 0., 1., 0.]))])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        model.eval()  # Disable dropout for this deterministic optimization check.
        initial = float(binary_losses(model(**model_inputs(batch, "cpu")), batch["labels"]).mean())
        for _ in range(60):
            loss = binary_losses(model(**model_inputs(batch, "cpu")), batch["labels"]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = float(binary_losses(model(**model_inputs(batch, "cpu")), batch["labels"]).mean())
        self.assertLess(final, initial * 0.3)


class R2DataTests(unittest.TestCase):
    def test_scene_split_and_rank_merge(self):
        rows = [dict(question_id=str(i), scene_id=f"scene{i//2}", candidates=["a", "b"]) for i in range(10)]
        train, dev = scene_split(rows)
        self.assertFalse({r["scene_id"] for r in train} & {r["scene_id"] for r in dev})
        shards = [{r["question_id"]: {"images": ["b", "a"], "scores": [0.8, 0.1]} for r in rows[j::2]} for j in range(2)]
        rankings, _ = merge_rankings(rows, shards)
        self.assertEqual(len(rankings), 10)
        with self.assertRaises(ValueError):
            merge_rankings(rows, shards[:1])
        with self.assertRaises(ValueError):
            merge_rankings(rows, shards + shards[:1])
        self.assertAlmostEqual(average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([1., 0., 1.])), 5/6)

    def test_text_prompt_exactly_matches_original(self):
        class Tokenizer:
            additional_special_tokens_ids = [1, 2]
            def __call__(self, text):
                return SimpleNamespace(input_ids=[ord(c) for c in text])
        source = ast.parse((REPO / "cdviews/train.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess_qwen")
        scope = dict(torch=torch, transformers=SimpleNamespace(PreTrainedTokenizer=object), Dict=Dict,
                     IGNORE_INDEX=-100)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "original", "exec"), scope)
        expected = scope["preprocess_qwen"]([{"from": "human", "value": "Where is it?"}, {"from": "gpt", "value": None}], Tokenizer())
        torch.testing.assert_close(text_token_ids(Tokenizer(), "Where is it?"), expected)

    def test_preparation_uses_train_labels_and_rejects_cache_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotation = root / "ScanQA/ScanQA_v1.0"
            annotation.mkdir(parents=True)
            write_json(annotation / "ScanQA_v1.0_train.json", [
                dict(question_id="q1", scene_id="s1", question="What?", answers=["secret"]),
                dict(question_id="q2", scene_id="s1", question="What else?", answers=["secret"])])
            images = root / "scannetv2/frames_square/s1/color"
            images.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                (images / name).touch()
            labels = root / "labels.json"
            write_json(labels, {"q1": {"Positive": ["a.jpg"], "Negative": ["b.jpg"]},
                                "q2": {"Positive": [], "Negative": ["b.jpg"]}})
            plan = build_plan(root, "train", labels)
            self.assertEqual(plan["excluded_ids"], ["q2"])
            self.assertNotIn("answers", plan["rows"][0])
            path = root / "plan.json"
            immutable_json(path, plan)
            immutable_json(path, plan)
            with self.assertRaises(ValueError):
                immutable_json(path, {**plan, "image_count": 99})


class R2TrainingTests(unittest.TestCase):
    def command(self, cache, run, epochs, ddp=False, resume=False):
        port = 0
        if ddp:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        prefix = [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
                  f"--master-port={port}", "--nproc_per_node=2",
                  "--module", "cdviews.train_r2"] if ddp else [sys.executable, "-m", "cdviews.train_r2"]
        cmd = prefix + ["--cache-dir", str(cache), "--run-dir", str(run), "--epochs", str(epochs),
                        "--batch-size", "2", "--accum-steps", "2", "--workers", "0", "--cpu",
                        "--embedding-dim", "8", "--output-dim", "16", "--num-heads", "2",
                        "--num-layers", "1", "--num-queries", "2", "--views-per-class", "2"]
        if resume:
            cmd.append("--resume")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if sys.platform == "darwin" and ddp:
            env["GLOO_SOCKET_IFNAME"] = "lo0"
        try:
            result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as error:
            self.fail(f"Training timeout: {error.stdout!r}\n{error.stderr!r}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_cache(root / "cache")
            self.command(root / "cache", root / "full", 2)
            self.command(root / "cache", root / "resume", 1)
            self.command(root / "cache", root / "resume", 2, resume=True)
            full = torch.load(root / "full/last.pt", weights_only=False)
            resumed = torch.load(root / "resume/last.pt", weights_only=False)
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], atol=0, rtol=0)
            split = read_json(root / "full/split.json")
            self.assertFalse(set(split["train_scenes"]) & set(split["dev_scenes"]))
            self.assertEqual(len(full["rng_by_rank"]), 1)

    def test_full_val_ranking_cli_uses_checkpoint_and_preserves_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_cache(root / "cache")
            train = read_json(root / "cache/train.json")
            text_folder = root / "cache/questions/val"
            text_folder.mkdir()
            rows = []
            for i in range(4675):
                qid = f"val-{i}"
                os.link(root / "cache/questions/train/q0_0.pt", text_folder / (qid + ".pt"))
                rows.append(dict(question_id=qid, scene_id="scene0", question="What?",
                                 candidates=["3.jpg", "1.jpg", "0.jpg", "2.jpg"]))
            write_json(root / "cache/val.json", dict(split="val", cache_id=train["cache_id"],
                       input_channel=8, rows=rows, selection_source_sha256="fixture"))
            model = QueryViewSelector(**tiny_config())
            torch.save(dict(architecture="r2_query_pool", model_config=tiny_config(), model=model.state_dict(),
                            cache_id=train["cache_id"], epoch=1, best_ap=0.5, smoke=False), root / "best.pt")
            result = subprocess.run([sys.executable, "-m", "cdviews.rank_r2", "--cpu",
                                     "--cache-dir", str(root / "cache"), "--checkpoint", str(root / "best.pt"),
                                     "--output-dir", str(root / "rank"), "--view-batch", "4"],
                                    cwd=REPO, env=dict(os.environ, OMP_NUM_THREADS="1"),
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            ranking = read_json(root / "rank/ScanQA_val_view_ranking.json")
            scores = read_json(root / "rank/ScanQA_val_view_ranking.json.scores.json")
            self.assertEqual(len(ranking), 4675)
            self.assertEqual(set(ranking["val-0"]), set(rows[0]["candidates"]))
            self.assertEqual(scores["val-0"], sorted(scores["val-0"], reverse=True))

    @unittest.skipUnless(os.environ.get("CDV_TEST_DDP") == "1", "Set CDV_TEST_DDP=1 for local two-process Gloo integration")
    def test_real_two_process_ddp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_cache(root / "cache")
            self.command(root / "cache", root / "ddp", 1, ddp=True)
            self.command(root / "cache", root / "ddp", 2, ddp=True, resume=True)
            checkpoint = torch.load(root / "ddp/last.pt", weights_only=False)
            config = read_json(root / "ddp/config.json")["signature"]
            self.assertEqual(config["effective_batch_questions"], 8)
            self.assertEqual(len(checkpoint["rng_by_rank"]), 2)
            self.assertEqual(checkpoint["epoch"], 2)
            log = [json.loads(line) for line in (root / "ddp/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["dev_questions"] for r in log], [2, 2])


if __name__ == "__main__":
    unittest.main()

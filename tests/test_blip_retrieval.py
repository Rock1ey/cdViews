"""CPU numerical/API checks using a tiny randomly initialized BLIP; no download."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from transformers import (BertTokenizer, BlipConfig, BlipForImageTextRetrieval,
                          BlipImageProcessor, BlipProcessor)

from cdviews.blip_retrieval import encode_images, encode_questions, rank_candidates, run_retrieval
from cdviews.soft_view_preparation import prepare_soft_views


def tiny_model():
    return BlipForImageTextRetrieval(BlipConfig(
        text_config=dict(vocab_size=12, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         encoder_hidden_size=16, max_position_embeddings=128),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8),
        image_text_hidden_size=8)).eval()


class RetrievalTests(unittest.TestCase):
    def test_cached_projection_matches_hf_cosine_head(self):
        torch.manual_seed(42)
        model = tiny_model()
        pixels = torch.randn(3, 3, 16, 16)
        tokens = {"input_ids": torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
        with torch.inference_mode():
            expected = model(pixel_values=pixels, **tokens, use_itm_head=False).itm_score
            actual = encode_images(model, pixels) @ encode_questions(model, tokens).T
        torch.testing.assert_close(actual, expected)

    def test_ranking_is_question_specific_and_masks_candidates(self):
        names = ["a.jpg", "b.jpg", "c.jpg"]
        images = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        first, scores = rank_candidates(names, images, torch.tensor([1., 0.]), names)
        self.assertEqual(first, ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(scores, [1., 0., -1.])
        second, _ = rank_candidates(names, images, torch.tensor([0., 1.]), names[::-1])
        self.assertEqual(second[0], "b.jpg")
        restricted, _ = rank_candidates(names, images, torch.tensor([1., 0.]), ["c.jpg", "b.jpg"])
        self.assertEqual(restricted, ["b.jpg", "c.jpg"])

    def test_offline_end_to_end_and_cached_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            vocab = model_dir / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhat\nchair\nroom\nis\nin\nthe\n?\n")
            tokenizer = BertTokenizer(vocab_file=str(vocab))
            processor = BlipProcessor(BlipImageProcessor(size={"height": 16, "width": 16}), tokenizer)
            processor.save_pretrained(model_dir)
            tiny_model().save_pretrained(model_dir, safe_serialization=False)
            images = root / "images/scene/color"
            images.mkdir(parents=True)
            for name, color in [("a.jpg", "red"), ("b.jpg", "blue"), ("c.jpg", "green")]:
                Image.new("RGB", (20, 20), color).save(images / name)
            annotation = root / "val.json"
            annotation.write_text(json.dumps([
                dict(question_id="q1", scene_id="scene", question="what is in the room?", answers=["SECRET"]),
                dict(question_id="q2", scene_id="scene", question="what chair?", answers=["SECRET"])]))
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps({"q1": ["c.jpg", "a.jpg", "b.jpg"], "q2": ["b.jpg", "a.jpg"]}))
            cfg = dict(dataset="ScanQA", inference_splits=["val"], seed=42, blip_model=str(model_dir),
                       retrieval_cache_folder=str(root / "cache"), ScanQA_file={"val": str(annotation)},
                       ranked_view_file=str(candidates), retrieval_ranked_view_file=str(root / "rank1.json"),
                       retrieval_batch_size=2, image_folder=str(root / "images"))
            with patch("torch.cuda.is_available", return_value=False), redirect_stdout(io.StringIO()):
                run_retrieval(cfg)
            first = json.loads((root / "rank1.json").read_text())
            self.assertEqual(set(first["q1"]), {"a.jpg", "b.jpg", "c.jpg"})
            self.assertEqual(set(first["q2"]), {"a.jpg", "b.jpg"})
            metadata = json.loads((root / "rank1.json.metadata.json").read_text())
            self.assertEqual(metadata["questions"], 2)
            self.assertFalse(metadata["nms"])
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)

            # A changed reference answer must not affect retrieval, and the
            # second run must read cached vectors without encoding images again.
            rows = json.loads(annotation.read_text())
            for row in rows:
                row["answers"] = ["CHANGED"]
            annotation.write_text(json.dumps(rows))
            cfg["retrieval_ranked_view_file"] = str(root / "rank2.json")
            with patch("torch.cuda.is_available", return_value=False), \
                 patch("cdviews.blip_retrieval.encode_images", side_effect=AssertionError("cache missed")), \
                 redirect_stdout(io.StringIO()):
                run_retrieval(cfg)
            self.assertEqual(first, json.loads((root / "rank2.json").read_text()))

            # Soft selection must reuse the retrieval image cache, retain the
            # author's per-question order as its prior, and never encode text.
            import numpy as np
            poses = root / "images/scene/pose"
            poses.mkdir()
            for index, name in enumerate(("a", "b", "c")):
                pose = np.eye(4)
                pose[0, 3] = index * 0.2
                np.savetxt(poses / f"{name}.txt", pose, delimiter=" ")
            cfg.update(soft_selected_view_file=str(root / "soft1.json"), input_views=2,
                       view_distance_folder=str(root / "distances"),
                       soft_penalty=0.5, soft_rank_scale=10., soft_pose_scale=0.5)
            with patch("torch.cuda.is_available", return_value=False), \
                 patch("cdviews.blip_retrieval.encode_images", side_effect=AssertionError("cache missed")), \
                 patch("cdviews.blip_retrieval.encode_questions", side_effect=AssertionError("text used")), \
                 redirect_stdout(io.StringIO()):
                prepare_soft_views(cfg)
            soft = json.loads((root / "soft1.json").read_text())
            self.assertEqual(soft["q1"][0], "c.jpg")
            self.assertEqual(soft["q2"][0], "b.jpg")
            self.assertEqual(len(set(soft["q1"])), 2)
            self.assertEqual(set(soft["q2"]), {"a.jpg", "b.jpg"})
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)
            meta = json.loads((root / "soft1.json.metadata.json").read_text())
            self.assertFalse(meta["answers_used"])
            self.assertFalse(meta["hard_nms"])
            self.assertEqual(meta["model_signature"], metadata["model_signature"])
            for row in rows:
                row["answers"] = ["DIFFERENT AGAIN"]
            annotation.write_text(json.dumps(rows))
            cfg["soft_selected_view_file"] = str(root / "soft2.json")
            # Exercise the standalone snapshot imports used by the launcher,
            # not only the package-import path used by this test process.
            snapshot = root / "snapshot"
            snapshot.mkdir()
            source = Path(__file__).resolve().parents[1] / "cdviews"
            for name in ("soft_view_preparation.py", "soft_selection.py", "blip_retrieval.py",
                         "view_distance_calculation.py", "qa_utils.py"):
                (snapshot / name).write_bytes((source / name).read_bytes())
            config = snapshot / "QA.yaml"
            config.write_text(yaml.safe_dump(cfg))
            result = subprocess.run(
                [sys.executable, str(snapshot / "soft_view_preparation.py"), "--cfg_file", str(config)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(soft, json.loads((root / "soft2.json").read_text()))
            self.assertTrue((root / "soft2.json.trace.json").is_file())
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)
            with self.assertRaises(FileExistsError):
                prepare_soft_views(cfg)


if __name__ == "__main__":
    unittest.main()

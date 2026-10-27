"""CPU numerical/API checks using a tiny randomly initialized BLIP; no download."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from transformers import (BertTokenizer, BlipConfig, BlipForImageTextRetrieval,
                          BlipImageProcessor, BlipProcessor)

from cdviews.blip_retrieval import encode_images, encode_questions, rank_candidates, run_retrieval


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


if __name__ == "__main__":
    unittest.main()

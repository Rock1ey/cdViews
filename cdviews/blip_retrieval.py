"""Offline BLIP ViT-L image/text cosine ranking before LVLM question answering.

The paper specifies BLIP ViT-L but does not publish the exact retrieval recipe.
We use Salesforce/blip-itm-large-coco and the projected CLS cosine route also
used in BridgeQA's eval_scene_best_views.py. No answers or captions enter BLIP.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor, __version__ as transformers_version
import yaml


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def encode_images(model, pixel_values):
    # [B, patches+1, hidden] -> normalized [B, retrieval_dim].
    tokens = model.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state
    return F.normalize(model.vision_proj(tokens[:, 0, :]), dim=-1)


def encode_questions(model, tokens):
    hidden = model.text_encoder(input_ids=tokens["input_ids"],
                                attention_mask=tokens["attention_mask"],
                                return_dict=True).last_hidden_state
    return F.normalize(model.text_proj(hidden[:, 0, :]), dim=-1)


def rank_candidates(image_names, image_features, question_feature, candidates):
    # Restrict each question to exactly its original candidate set. A scene can
    # contain different question-specific pools; their union is only a cache.
    indices = {name: i for i, name in enumerate(image_names)}
    names = sorted(candidates)
    features = image_features[[indices[name] for name in names]]
    scores = (features @ question_feature).tolist()
    # Filename breaks ties, without using the author's relevance ranking.
    order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i]))
    return [names[i] for i in order], [scores[i] for i in order]


def scene_features(model, processor, folder, names, cache_dir, model_signature, batch_size, device):
    image_identity = []
    for name in names:
        stat = (folder / name).stat()
        image_identity.append((name, stat.st_size, stat.st_mtime_ns))
    key = fingerprint({"model": model_signature, "folder": str(folder.resolve()),
                       "images": image_identity})
    cache_file = cache_dir / f"{key}.pt"
    if cache_file.exists():
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    features = []
    for start in range(0, len(names), batch_size):
        images = []
        for name in names[start:start + batch_size]:
            with Image.open(folder / name) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        features.append(encode_images(model, pixels).cpu())
    features = torch.cat(features)
    temporary = cache_file.with_suffix(f".{os.getpid()}.tmp")
    torch.save(features, temporary)
    temporary.replace(cache_file)
    return features


def run_retrieval(cfg):
    if cfg["dataset"] != "ScanQA" or cfg["inference_splits"] != ["val"]:
        raise ValueError("This retrieval experiment is restricted to ScanQA val")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model_dir = Path(cfg["blip_model"])
    # Hash the checkpoint/config once; cache keys cannot silently reuse another
    # checkpoint, preprocessing configuration or Transformers implementation.
    model_files = ["pytorch_model.bin", "config.json", "preprocessor_config.json",
                   "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    hashes = {name: file_sha256(model_dir / name) for name in model_files}
    signature = fingerprint({"files": hashes, "transformers": transformers_version,
                             "torch": torch.__version__, "dtype": "float32",
                             "implementation": file_sha256(__file__)})
    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    model = BlipForImageTextRetrieval.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=False,
        torch_dtype=torch.float32).eval().to(device)
    cache_dir = Path(cfg["retrieval_cache_folder"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(cfg["ScanQA_file"]["val"]).read_text())
    candidate_path = Path(cfg["ranked_view_file"].format("ScanQA", "val"))
    candidates = json.loads(candidate_path.read_text())
    output = Path(cfg["retrieval_ranked_view_file"].format("ScanQA", "val"))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retrieval ranking: {output}")
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(row)
    rankings, score_rows = {}, {}
    batch_size = cfg["retrieval_batch_size"]

    with torch.inference_mode():
        for index, (scene, questions) in enumerate(scenes.items(), 1):
            names = sorted({name for row in questions for name in candidates[str(row["question_id"])]})
            folder = Path(cfg["image_folder"]) / scene / "color"
            image_features = scene_features(model, processor, folder, names, cache_dir,
                                            signature, batch_size, device)
            for start in range(0, len(questions), batch_size):
                batch = questions[start:start + batch_size]
                tokens = processor.tokenizer([row["question"] for row in batch],
                                             padding="max_length", truncation=True,
                                             max_length=70, return_tensors="pt").to(device)
                question_features = encode_questions(model, tokens).cpu()
                for row, feature in zip(batch, question_features):
                    qid = str(row["question_id"])
                    rankings[qid], score_rows[qid] = rank_candidates(
                        names, image_features, feature, candidates[qid])
            print(f"BLIP scene {index}/{len(scenes)}: {scene}, {len(names)} images, "
                  f"{len(questions)} questions", flush=True)

    output.write_text(json.dumps(rankings))
    Path(str(output) + ".scores.json").write_text(json.dumps(score_rows))
    metadata = {"model_dir": str(model_dir), "model_files_sha256": hashes,
                "model_signature": signature, "transformers": transformers_version,
                "torch": torch.__version__, "device": str(device), "dtype": "float32",
                "score": "cosine(normalized projected image CLS, normalized projected question CLS)",
                "text": "original question only; no reference answers", "text_max_length": 70,
                "preprocessor": processor.image_processor.to_dict(), "nms": False,
                "questions": len(rankings), "split": "val",
                "candidates_sha256": file_sha256(candidate_path),
                "ranking_sha256": file_sha256(output)}
    Path(str(output) + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(rankings)} BLIP rankings to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", type=Path, required=True)
    args = parser.parse_args()
    run_retrieval(yaml.safe_load(args.cfg_file.read_text()))

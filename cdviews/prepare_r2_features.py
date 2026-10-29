"""Cache frozen LLaVA patch/text embeddings; torchrun shards scenes over GPUs."""

import argparse
from collections import defaultdict
import json
from importlib.metadata import version
from pathlib import Path

import torch

from .r2_data import read_json, write_json, sha256, fingerprint
from .r2_runtime import distributed_device, barrier, atomic_save


def text_token_ids(tokenizer, question):
    # Exactly the text-only Qwen prompt used by the upstream selector.
    start, end = tokenizer.additional_special_tokens_ids
    tok = lambda text: tokenizer(text).input_ids
    nl = tok("\n")
    ids = [start] + tok("system") + nl + tok("You are a helpful assistant.") + [end] + nl
    ids += tok("<|im_start|>user") + nl + tok(question) + [end] + nl
    ids += tok("<|im_start|>assistant") + nl
    return torch.tensor([ids], dtype=torch.long)


def build_plan(root, split, labels_path):
    annotation = root / "ScanQA/ScanQA_v1.0" / f"ScanQA_v1.0_{split}.json"
    rows = read_json(annotation)
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {annotation}")
    if split == "val" and len(rows) != 4675:
        raise ValueError("ScanQA val must contain exactly 4675 questions")
    source = labels_path if split == "train" else root / "ScanQA_val_view_ranking.json"
    supplied = read_json(source)
    prepared, excluded = [], []
    scene_images = defaultdict(set)
    for row, qid in zip(rows, ids):
        item = dict(question_id=qid, scene_id=row["scene_id"], question=row["question"])
        if split == "train":
            labeling = supplied.get(qid, {})
            positive = sorted(set(labeling.get("Positive", [])))
            negative = sorted(set(labeling.get("Negative", [])))
            if set(positive) & set(negative):
                raise ValueError(f"Conflicting view labels for {qid}")
            if not positive or not negative:
                excluded.append(qid)
                continue
            item.update(positive=positive, negative=negative)
            names = positive + negative
        else:
            names = supplied[qid]
            if not names or len(names) != len(set(names)):
                raise ValueError(f"Invalid val candidates for {qid}")
            item["candidates"] = names
        scene_images[row["scene_id"]].update(names)
        prepared.append(item)
    if not prepared:
        raise ValueError(f"No usable questions for {split}")
    image_stats = []
    for scene, names in sorted(scene_images.items()):
        for name in sorted(names):
            path = root / "scannetv2/frames_square" / scene / "color" / name
            stat = path.stat()  # Missing frames fail before loading a model.
            image_stats.append((scene, name, stat.st_size, stat.st_mtime_ns))
    return dict(split=split, rows=prepared, excluded_ids=excluded,
                annotations_sha256=sha256(annotation), selection_source_sha256=sha256(source),
                image_identity=fingerprint(image_stats), image_count=len(image_stats),
                scene_images={scene: sorted(names) for scene, names in sorted(scene_images.items())})


def cache_spec(root, model_path):
    weights = sorted(list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin")))
    if not weights:
        raise FileNotFoundError(f"No local LLaVA weights in {model_path}")
    metadata = {path.name: sha256(path) for path in sorted(model_path.iterdir())
                if path.suffix in (".json", ".model")}
    return dict(format_version=1, data_root=str(root), model_path=str(model_path),
                config_sha256=metadata,
                weight_identity=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights],
                recipe_sha256=sha256(__file__), dtype="float16", image_newline=True,
                dependencies={"torch": torch.__version__, "transformers": version("transformers"),
                              "pillow": version("Pillow")},
                text="upstream Qwen text-only prompt; frozen token embeddings")


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f"Cache inputs/config changed; use a new cache directory: {path}")
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.splits)) != len(args.splits):
        parser.error("Positive batch size and distinct splits required")
    root, cache = args.data_root.resolve(), args.cache_dir.resolve()
    model_path = (args.model_path or root / "model/llava-onevision-qwen2-7b-ov").resolve()
    labels = (args.labels or root / "ScanQA_view_labeling.json").resolve()
    rank, world, device = distributed_device()
    spec = cache_spec(root, model_path)
    plans = {split: build_plan(root, split, labels) for split in args.splits}
    if rank == 0:
        for split, plan in plans.items():
            print(f"{split}: {len(plan['rows'])} questions, {len(plan['excluded_ids'])} excluded, "
                  f"{plan['image_count']} images; approx {plan['image_count'] * 730 * 3584 * 2 / 2**30:.1f} GiB "
                  "at 730x3584 fp16 (verify actual model shape)", flush=True)
    if args.check_only:
        return
    if device.type != "cuda":
        raise RuntimeError("Frozen LLaVA feature extraction requires a CUDA server")
    if rank == 0:
        cache.mkdir(parents=True, exist_ok=True)
        immutable_json(cache / "config.json", spec)
        for split, plan in plans.items():
            immutable_json(cache / f"{split}.plan.json", plan)
    barrier()

    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    tokenizer, model, processor, _ = load_pretrained_model(
        str(model_path), None, get_model_name_from_path(str(model_path)), device_map={"": str(device)})
    model.eval().requires_grad_(False)
    dimension = model.get_model().embed_tokens.weight.shape[1]
    cache_id = fingerprint(spec)
    with torch.inference_mode():
        for split, plan in plans.items():
            scene_questions = defaultdict(list)
            for row in plan["rows"]:
                scene_questions[row["scene_id"]].append(row)
            scenes = sorted(scene_questions)
            for scene in scenes[rank::world]:
                folder = cache / "images" / scene
                folder.mkdir(parents=True, exist_ok=True)
                names = [name for name in plan["scene_images"][scene] if not (folder / (name + ".pt")).exists()]
                for start in range(0, len(names), args.batch_size):
                    batch = names[start:start + args.batch_size]
                    pixels = []
                    for name in batch:
                        with Image.open(root / "scannetv2/frames_square" / scene / "color" / name) as image:
                            pixels.append(processor.preprocess(image.convert("RGB"), return_tensors="pt")["pixel_values"])
                    features = model.encode_images(torch.cat(pixels).to(device=device, dtype=torch.float16))
                    newline = model.get_model().image_newline[None, None].expand(len(batch), 1, -1)
                    features = torch.cat((features, newline), dim=1).half().cpu()
                    for name, feature in zip(batch, features):
                        atomic_save(feature.clone(), folder / (name + ".pt"))
                question_folder = cache / "questions" / split
                question_folder.mkdir(parents=True, exist_ok=True)
                for row in scene_questions[scene]:
                    output = question_folder / (row["question_id"] + ".pt")
                    if not output.exists():
                        ids = text_token_ids(tokenizer, row["question"])[0]
                        embedding = model.get_model().embed_tokens(ids.to(device)).half().cpu()
                        atomic_save(embedding, output)
                print(f"rank={rank} {split} finished {scene}", flush=True)
            barrier()
            if rank == 0:
                # Only publish a complete manifest after all ranks succeeded.
                manifest = {**plan, "cache_id": cache_id, "input_channel": dimension}
                immutable_json(cache / f"{split}.json", manifest)
                print(f"Ready: {cache / (split + '.json')}", flush=True)
            barrier()


if __name__ == "__main__":
    main()

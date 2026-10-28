"""Prepare soft-selected views before loading LLaVA; no training or QA labels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

# Script execution uses the experiment's sibling snapshots. Package imports are
# for CPU tests and interactive use from the repository.
if __package__:
    from . import blip_retrieval as blip
    from .soft_selection import soft_select_views, question_coverage_weights
    from .view_distance_calculation import calculate_view_distance
else:
    import blip_retrieval as blip
    from soft_selection import soft_select_views, question_coverage_weights
    from view_distance_calculation import calculate_view_distance


def prepare_soft_views(cfg):
    if cfg["dataset"] != "ScanQA" or cfg["inference_splits"] != ["val"]:
        raise ValueError("Soft selection supports only ScanQA val")
    output = Path(cfg["soft_selected_view_file"].format("ScanQA", "val"))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selected views: {output}")
    method = cfg.get("view_selection", "soft")
    coverage_weight = cfg.get("coverage_weight", 0.5) if method == "coverage" else 0.0
    use_coverage = coverage_weight > 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model_dir = Path(cfg["blip_model"])
    model_files = ["pytorch_model.bin", "config.json", "preprocessor_config.json",
                   "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    hashes = {name: blip.file_sha256(model_dir / name) for name in model_files}
    # Match the existing retrieval cache key exactly to reuse its image features.
    signature = blip.fingerprint({"files": hashes, "transformers": blip.transformers_version,
                                  "torch": torch.__version__, "dtype": "float32",
                                  "implementation": blip.file_sha256(blip.__file__)})
    processor = blip.BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    model = blip.BlipForImageTextRetrieval.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=False,
        torch_dtype=torch.float32).eval().to(device)
    cache_dir = Path(cfg["retrieval_cache_folder"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    distance_dir = Path(cfg["view_distance_folder"])
    distance_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(cfg["ScanQA_file"]["val"]).read_text())
    questions = {str(row["question_id"]): row["question"] for row in rows}
    ranking_path = Path(cfg["ranked_view_file"].format("ScanQA", "val"))
    rankings = json.loads(ranking_path.read_text())
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(str(row["question_id"]))
    selected, traces, pose_hashes, coverage_inputs = {}, {}, {}, {}
    params = {"penalty": cfg["soft_penalty"], "rank_scale": cfg["soft_rank_scale"],
              "pose_scale": cfg["soft_pose_scale"]}

    with torch.inference_mode():
        for index, (scene, question_ids) in enumerate(scenes.items(), 1):
            names = sorted({name for qid in question_ids for name in rankings[qid]})
            folder = Path(cfg["image_folder"]) / scene / "color"
            features = blip.scene_features(model, processor, folder, names, cache_dir,
                                           signature, cfg["retrieval_batch_size"], device).numpy()
            name_to_index = {name: i for i, name in enumerate(names)}
            if use_coverage:
                # Cache the scene kernel in memory, independent of question.
                # Fourth power reduces the influence of weak global similarity.
                scene_kernel = np.clip(features @ features.T, 0.0, 1.0) ** 4
                question_vectors = {}
                batch_size = cfg["retrieval_batch_size"]
                for start in range(0, len(question_ids), batch_size):
                    ids = question_ids[start:start + batch_size]
                    tokens = processor.tokenizer([questions[qid] for qid in ids],
                                                 padding="max_length", truncation=True,
                                                 max_length=70, return_tensors="pt").to(device)
                    vectors = blip.encode_questions(model, tokens).cpu().numpy()
                    question_vectors.update(zip(ids, vectors))
            distance_path = distance_dir / f"{scene}.csv"
            if not distance_path.exists():
                distances = calculate_view_distance(scene, SimpleNamespace(**cfg))
                distances.to_csv(distance_path)
            distances = pd.read_csv(distance_path, index_col=0)
            aligned_distances = distances.loc[names, names].to_numpy()
            if not np.isfinite(aligned_distances).all():
                raise ValueError(f"Invalid pose distances for {scene}")
            pose_hashes[scene] = blip.file_sha256(distance_path)
            for qid in question_ids:
                candidates = rankings[qid]
                indices = [name_to_index[name] for name in candidates]
                coverage_args = {}
                if use_coverage:
                    weights, similarities = question_coverage_weights(
                        features[indices], question_vectors[qid], rank_scale=cfg["soft_rank_scale"],
                        temperature=cfg["coverage_temperature"], text_weight=cfg["coverage_text_weight"])
                    coverage_args = {"coverage_kernel": scene_kernel[np.ix_(indices, indices)],
                                     "coverage_weights": weights, "coverage_weight": coverage_weight}
                    coverage_inputs[qid] = {"images": candidates, "weights": weights.tolist(),
                                            "question_cosines": similarities.tolist()}
                selected[qid], traces[qid] = soft_select_views(
                    candidates, features[indices], aligned_distances[np.ix_(indices, indices)],
                    cfg["input_views"], **params, **coverage_args)
            print(f"{method} selection scene {index}/{len(scenes)}: {scene}, "
                  f"{len(question_ids)} questions", flush=True)

    output.write_text(json.dumps(selected))
    Path(str(output) + ".trace.json").write_text(json.dumps(traces, indent=2))
    if use_coverage:
        Path(str(output) + ".coverage.json").write_text(json.dumps(coverage_inputs))
    metadata = {"method": method, "version": 2 if method == "coverage" else 1,
                "split": "val", "questions": len(selected),
                "input_views": cfg["input_views"], "parameters": params,
                "relevance": "1 / (1 + zero_based_cdviews_rank / rank_scale); not raw selector scores",
                "redundancy": "max_selected(clip(BLIP_image_cosine,0,1) * (0.5+0.5*exp(-pose_distance/pose_scale)))",
                "objective": "relevance - penalty * redundancy", "hard_nms": False,
                "answers_used": False, "model_dir": str(model_dir), "model_files_sha256": hashes,
                "model_signature": signature, "transformers": blip.transformers_version,
                "torch": torch.__version__, "device": str(device),
                "preprocessor": processor.image_processor.to_dict(),
                "ranking_sha256": blip.file_sha256(ranking_path), "pose_distance_sha256": pose_hashes,
                "selection_sha256": blip.file_sha256(output)}
    if method == "coverage":
        metadata["objective"] = "rank_relevance - penalty * redundancy + coverage_weight * marginal_coverage"
        metadata["coverage"] = {
            "weight": coverage_weight, "temperature": cfg["coverage_temperature"],
            "text_weight": cfg["coverage_text_weight"], "kernel": "clip(image_cosine,0,1)^4",
            "weights": "normalize(rank_prior * exp(text_weight * question_cosine / temperature))",
            "set_function": "F(S) = sum_u weights[u] * max_{j in S} kernel[u,j]",
            "anchor": "original cdViews top-1; no coverage bonus on first step",
            "text_max_length": 70, "active": use_coverage,
            "question_sha256": blip.fingerprint(questions),
            "interpretation": "feature-space coverage proxy; not explicit object/relation evidence"}
    Path(str(output) + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved soft selections for {len(selected)} questions to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_file", type=Path, required=True)
    args = parser.parse_args()
    prepare_soft_views(yaml.safe_load(args.cfg_file.read_text()))

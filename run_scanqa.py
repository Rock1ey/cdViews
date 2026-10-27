"""Run ScanQA validation on one GPU by default, then validate and merge answers.

Use --stage smoke before --stage full. Each stage needs a new output directory;
the upstream inference script saves only at the end of each split, not per question.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


SPLITS = ("test_w_obj", "test_wo_obj")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def merge_answers(stage_dir, datasets, subsets, gpus):
    for split in datasets:
        by_id = {}
        views_by_id = {}
        expected_ids = set()
        for gpu in gpus:
            rows = subsets[gpu][split]
            expected = {str(row["question_id"]): row["scene_id"] for row in rows}
            answers = read_json(stage_dir / f"gpu{gpu}" / f"ScanQA_{split}_answers.json")
            found = set()
            for answer in answers:
                qid = str(answer["question_id"])
                if qid in found or qid in by_id:
                    raise ValueError(f"Duplicate answer: {split}/{qid}")
                if qid not in expected or answer["scene_id"] != expected[qid]:
                    raise ValueError(f"Unexpected question or scene: {split}/{qid}")
                if not isinstance(answer.get("answer_top10"), list) or not answer["answer_top10"]:
                    raise ValueError(f"Missing answer list: {split}/{qid}")
                found.add(qid)
                by_id[qid] = answer
            if found != set(expected):
                raise ValueError(f"Incomplete answers: GPU {gpu}, {split}, {len(found)}/{len(expected)}")
            trace = read_json(stage_dir / f"gpu{gpu}" / f"ScanQA_{split}_answers.json.views.json")
            trace_ids = [row["question_id"] for row in trace["questions"]]
            if len(set(trace_ids)) != len(trace_ids) or set(trace_ids) != found:
                raise ValueError(f"Selected-view trace does not match answers: GPU {gpu}, {split}")
            views_by_id.update({row["question_id"]: row for row in trace["questions"]})
            expected_ids.update(expected)
        merged = [by_id[str(row["question_id"])] for row in datasets[split]
                  if str(row["question_id"]) in expected_ids]
        output = stage_dir / f"ScanQA_{split}_answers.json"
        write_json(output, merged)
        trace["questions"] = [views_by_id[row["question_id"]] for row in merged]
        write_json(Path(str(output) + ".views.json"), trace)
        print(f"[PASS] {split}: merged {len(merged)} answers -> {output}", flush=True)


def run_jobs(jobs, repo, log_name):
    """Finish one model phase before loading the next model on the same GPU."""
    processes = []
    try:
        for gpu, worker_dir, env, command in jobs:
            log_path = worker_dir / log_name
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((gpu, process))
            print(f"GPU {gpu}: PID={process.pid}; log: {log_path}", flush=True)
        pending = dict(processes)
        while pending:
            for gpu, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    print(f"GPU {gpu}: exit={code}", flush=True)
                    del pending[gpu]
                    if code != 0:
                        raise SystemExit(f"GPU {gpu} failed. Read its {log_name}; later phases were not started.")
            if pending:
                time.sleep(1)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ["CDV_DATA"]) if os.environ.get("CDV_DATA") else None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--gpus", nargs="+", default=[os.environ.get("CDV_GPU", "0")],
                        help="Defaults to one GPU: CDV_GPU or 0; explicitly pass two IDs only when needed")
    parser.add_argument("--splits", nargs="+", choices=("val", *SPLITS), default=["val"],
                        help="Default: val (the project's evaluation split)")
    parser.add_argument("--view-selection", choices=("cdviews", "uniform", "retrieval"), default="cdviews",
                        help="Uniform: random sampling; retrieval: BLIP cosine top-k; both disable NMS")
    parser.add_argument("--input-views", type=int, default=None,
                        help="Default: 17 for retrieval, 9 for cdviews/uniform")
    parser.add_argument("--blip-model", type=Path, help="Local Salesforce/blip-itm-large-coco directory")
    parser.add_argument("--retrieval-batch-size", type=int, default=8,
                        help="BLIP encoding batch size; does not change the QA view budget")
    parser.add_argument("--seed", type=int, default=42,
                        help="Stable per-question view sampling and decoding seed")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Use 0 for greedy decoding; default 0.2 retains the previous sampling setting")
    args = parser.parse_args()
    if args.input_views is None:
        args.input_views = 17 if args.view_selection == "retrieval" else 9
    if args.data_root is None:
        parser.error("Set CDV_DATA (source scripts/activate.sh lab|cloud) or pass --data-root")
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error("Choose one GPU identifier or two different GPU identifiers")
    if len(set(args.splits)) != len(args.splits):
        parser.error("Do not repeat split names")
    if args.input_views < 1 or args.temperature < 0:
        parser.error("--input-views must be positive and --temperature must be nonnegative")
    if args.retrieval_batch_size < 1:
        parser.error("--retrieval-batch-size must be positive")
    if args.view_selection == "retrieval" and args.splits != ["val"]:
        parser.error("The retrieval experiment supports only --splits val")

    repo = args.repo.resolve()
    root = args.data_root.resolve()
    blip_model = (args.blip_model or root / "model/blip-itm-large-coco").resolve()
    if args.view_selection == "retrieval":
        for name in ("pytorch_model.bin", "config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
            if not (blip_model / name).is_file():
                raise FileNotFoundError(f"Missing BLIP file: {blip_model / name}. Run scripts/download_blip.py first.")
    stage_dir = args.run_dir.resolve() / args.stage
    entry = repo / "cdviews/qa_inference.py"
    if not entry.is_file():
        raise FileNotFoundError(f"Not a cdViews repository: {repo}")
    base = yaml.safe_load((repo / "cfgs/QA.yaml").read_text())
    annotations = root / "ScanQA/ScanQA_v1.0"
    datasets = {split: read_json(annotations / f"ScanQA_v1.0_{split}.json") for split in args.splits}

    # The upstream script hardcodes the test splits. For validation, create an
    # inspectable copy changing only that selection; leave the original untouched.
    selected_source = None
    if args.splits != list(SPLITS):
        source = entry.read_text()
        original = "test_mode = ['test_w_obj', 'test_wo_obj'] if args.dataset == 'ScanQA' else ['test', ]"
        if source.count(original) != 1:
            raise ValueError("The upstream split-selection line changed; inspect qa_inference.py before adapting it")
        selected_source = source.replace(original, f"test_mode = {args.splits!r}")

    # Both methods use exactly the same candidate sets. Uniform ignores their
    # ranks, rather than sampling only the selector's top-k or NMS survivors.
    rankings = {}
    image_names = {}
    for split, rows in datasets.items():
        if split == "val" and len(rows) != 4675:
            raise ValueError(f"Expected full ScanQA val annotations (4675 questions), found {len(rows)}")
        if split == "val" and any(not row.get("answers") for row in rows):
            raise ValueError("ScanQA val must contain reference answers")
        ids = [str(row["question_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate question IDs: {split}")
        ranking = read_json(root / f"ScanQA_{split}_view_ranking.json")
        rankings[split] = ranking
        for row, qid in zip(rows, ids):
            frames = ranking.get(qid)
            if not isinstance(frames, list) or not frames or not all(isinstance(x, str) for x in frames):
                raise ValueError(f"Missing/invalid flat ranking: {split}/{qid}")
            if len(set(frames)) != len(frames):
                raise ValueError(f"Duplicate candidate frames: {split}/{qid}")
            scene = row["scene_id"]
            if scene not in image_names:
                image_names[scene] = set(os.listdir(root / "scannetv2/frames_square" / scene / "color"))
            missing = set(frames) - image_names[scene]
            if missing:
                raise FileNotFoundError(f"Missing candidate images for {split}/{qid}: {sorted(missing)[:5]}")

    # Keep the original question order within each disjoint shard.
    subsets = {}
    for worker, gpu in enumerate(args.gpus):
        subsets[gpu] = {}
        for split, rows in datasets.items():
            shard = rows[worker::len(args.gpus)]
            subsets[gpu][split] = shard[:2] if args.stage == "smoke" else shard

    # Refuse to overwrite an earlier run, including a partially completed one.
    stage_dir.mkdir(parents=True, exist_ok=False)
    if selected_source is not None:
        entry = stage_dir / "qa_inference_selected_splits.py"
        entry.write_text(selected_source)
    # Keep the sampler next to the inference snapshot, so its implementation can
    # be inspected together with the selected-view trace after the experiment.
    (stage_dir / "view_sampling.py").write_text((repo / "cdviews/view_sampling.py").read_text())
    retrieval_entry = stage_dir / "blip_retrieval.py"
    if args.view_selection == "retrieval":
        retrieval_entry.write_text((repo / "cdviews/blip_retrieval.py").read_text())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True)
    (stage_dir / "code.diff").write_text(diff.stdout)
    inputs = {}
    for split in args.splits:
        for name, path in [("annotations", annotations / f"ScanQA_v1.0_{split}.json"),
                           ("candidates", root / f"ScanQA_{split}_view_ranking.json")]:
            inputs[f"{split}_{name}"] = {"path": str(path),
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(stage_dir / "experiment.json", {
        "method": args.view_selection, "input_views": args.input_views,
        "seed": args.seed, "temperature": args.temperature, "splits": args.splits,
        "stage": args.stage, "gpus": args.gpus, "git_commit": revision.stdout.strip(),
        "candidate_pool": "All entries in the supplied ranking, ignoring rank for uniform",
        "uniform_order": "Random draw order; sampling without replacement",
        "inputs": inputs,
        "retrieval": {"blip_model": str(blip_model), "batch_size": args.retrieval_batch_size,
                      "score": "projected CLS cosine", "nms": False}
                     if args.view_selection == "retrieval" else None,
    })
    jobs = []
    for gpu in args.gpus:
        worker_dir = stage_dir / f"gpu{gpu}"
        worker_dir.mkdir()
        distance_dir = worker_dir / "view_distance"
        distance_dir.mkdir()
        cfg = copy.deepcopy(base)
        cfg["ScanQA_file"] = {
            split: str(annotations / f"ScanQA_v1.0_{split}.json")
            for split in ("train", "val", *SPLITS)
        }
        for split, rows in subsets[gpu].items():
            path = worker_dir / f"ScanQA_v1.0_{split}.json"
            write_json(path, rows)
            cfg["ScanQA_file"][split] = str(path)
            write_json(worker_dir / f"ScanQA_{split}_view_ranking.json",
                       {str(row["question_id"]): rankings[split][str(row["question_id"])] for row in rows})
        cfg.update({
            "dataset": "ScanQA",
            "input_views": args.input_views,
            "view_selection": args.view_selection,
            "inference_splits": args.splits,
            "seed": args.seed,
            "batch_size": 1,
            "temperature": args.temperature,
            "top_p": None,
            "num_beams": 1,
            "image_folder": str(root / "scannetv2/frames_square"),
            "feature_folder": str(root / "scannetv2/llava_ov_feats"),
            "view_distance_folder": str(distance_dir),
            "LVLM_ckpt": str(root / "model/llava-onevision-qwen2-7b-ov"),
            "pretrained_view_selector_ckpt": str(root / "model/{}_view_selector.pth"),
            "ranked_view_file": str(worker_dir / "{}_{}_view_ranking.json"),
            "retrieval_ranked_view_file": str(worker_dir / "{}_{}_blip_ranking.json"),
            "blip_model": str(blip_model),
            "retrieval_batch_size": args.retrieval_batch_size,
            "retrieval_cache_folder": str(root / "scannetv2/blip_itc_cache"),
            "answers_file": str(worker_dir / "{}_{}_answers.json"),
            "view_labeling_file": str(root / "{}_view_labeling.json"),
            "caption_file": str(root / "{}_QA_captions.json"),
            "work_dir": str(worker_dir),
        })
        config_path = worker_dir / "QA.yaml"
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": os.pathsep.join(filter(None, [str(repo / "cdviews"), str(repo), env.get("PYTHONPATH")])),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "HF_HOME": str(root / "hf_cache"),
            "HF_HUB_CACHE": str(root / "hf_cache/hub"),
            "HUGGINGFACE_HUB_CACHE": str(root / "hf_cache/hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        })
        # Override an older Transformers cache setting if inherited from the shell.
        env.pop("TRANSFORMERS_CACHE", None)
        command = [sys.executable, "-u", str(entry), "--cfg_file", str(config_path)]
        jobs.append((gpu, worker_dir, env, command))
        counts = {split: len(rows) for split, rows in subsets[gpu].items()}
        print(f"GPU {gpu}: {counts}; config: {config_path}", flush=True)

    print(f"Method={args.view_selection}; views={args.input_views}; seed={args.seed}; "
          f"temperature={args.temperature}. Hardware/backend differences can still affect outputs.", flush=True)
    if args.view_selection == "retrieval":
        retrieval_jobs = [(gpu, worker_dir, env,
                           [sys.executable, "-u", str(retrieval_entry), "--cfg_file", str(worker_dir / "QA.yaml")])
                          for gpu, worker_dir, env, _ in jobs]
        run_jobs(retrieval_jobs, repo, "retrieval.log")
    run_jobs(jobs, repo, "inference.log")
    merge_answers(stage_dir, datasets, subsets, args.gpus)


if __name__ == "__main__":
    main()

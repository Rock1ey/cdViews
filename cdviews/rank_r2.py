"""Score the original ScanQA val candidate pool with an explicitly chosen R2 checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

from .r2_model import QueryViewSelector
from .r2_data import FeatureStore, load_manifest, read_json, write_json, sha256, score_question
from .r2_runtime import distributed_device, barrier, code_record


def merge_rankings(rows, shards):
    expected = {row["question_id"]: row["candidates"] for row in rows}
    ranks, scores = {}, {}
    for shard in shards:
        for qid, result in shard.items():
            if qid in ranks or qid not in expected:
                raise ValueError(f"Duplicate or unexpected ranking: {qid}")
            names, values = result["images"], result["scores"]
            if (len(names) != len(set(names)) or set(names) != set(expected[qid])
                    or len(values) != len(names) or not torch.isfinite(torch.tensor(values)).all()):
                raise ValueError(f"Invalid ranking/candidate coverage: {qid}")
            ranks[qid], scores[qid] = names, values
    if set(ranks) != set(expected):
        raise ValueError("Incomplete ranking shards")
    return ({qid: ranks[qid] for qid in expected}, {qid: scores[qid] for qid in expected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-batch", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.view_batch < 1:
        parser.error("view-batch must be positive")
    rank, world, device = distributed_device(args.cpu)
    cache, output = args.cache_dir.resolve(), args.output_dir.resolve()
    manifest = load_manifest(cache, "val")
    rows = manifest["rows"]
    if len(rows) != 4675 or len({r["question_id"] for r in rows}) != 4675:
        raise ValueError("Ranking requires all 4675 distinct ScanQA val questions")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["architecture"] != "r2_query_pool" or state["cache_id"] != manifest["cache_id"]:
        raise ValueError("Checkpoint architecture or feature cache does not match")
    if state.get("smoke"):
        raise ValueError("Use a full-training checkpoint, not a smoke checkpoint, for val ranking")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        revision = code_record(output)
    barrier()
    model = QueryViewSelector(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    store = FeatureStore(cache, "val")
    scenes = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(row)
    results = {}
    for scene in sorted(scenes)[rank::world]:
        for row in scenes[scene]:
            names = row["candidates"]
            scores = score_question(model, store, row, names, device, args.view_batch).tolist()
            order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i]))
            results[row["question_id"]] = dict(images=[names[i] for i in order], scores=[scores[i] for i in order])
        print(f"rank={rank} scored {scene}: {len(scenes[scene])} questions", flush=True)
    write_json(output / f"rank{rank}.json", results)
    barrier()
    if rank == 0:
        rankings, scores = merge_rankings(rows, [read_json(output / f"rank{i}.json") for i in range(world)])
        target = output / "ScanQA_val_view_ranking.json"
        write_json(target, rankings)
        write_json(Path(str(target) + ".scores.json"), scores)
        write_json(Path(str(target) + ".metadata.json"), dict(
            architecture="r2_query_pool", split="val", questions=4675, model_config=state["model_config"],
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
            epoch=state["epoch"], best_internal_dev_ap=state["best_ap"], cache_id=manifest["cache_id"],
            val_manifest_sha256=sha256(cache / "val.json"),
            reference_ranking_sha256=manifest["selection_source_sha256"],
            ranking_sha256=sha256(target), score="raw cosine", git_commit=revision, world_size=world))
        print(f"Ready: {target}", flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

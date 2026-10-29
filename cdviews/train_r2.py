"""Train R2 from frozen features on one GPU or torchrun DDP; no official val use."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .r2_model import QueryViewSelector, binary_losses
from .r2_data import (QuestionDataset, FeatureStore, collate_questions, model_inputs,
                      score_question, average_precision, scene_split, load_manifest,
                      read_json, write_json, sha256)
from .r2_runtime import distributed_device, barrier, seed_everything, atomic_save, code_record


@torch.no_grad()
def validate(model, rows, cache, device, rank=0, world=1, view_batch=16):
    model.eval()
    store = FeatureStore(cache, "train")
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for row in rows[rank::world]:
        names = sorted(row["positive"] + row["negative"])
        positive = set(row["positive"])
        labels = torch.tensor([float(name in positive) for name in names])
        scores = score_question(model, store, row, names, device, view_batch)
        order = torch.argsort(scores, descending=True, stable=True)
        recall = float(labels[order[:9]].sum() / labels.sum())
        totals += torch.tensor([average_precision(scores, labels), recall,
                                float(binary_losses(scores, labels).mean()), 1], device=device)
    if world > 1:
        dist.all_reduce(totals)
    return dict(dev_ap=float(totals[0] / totals[3]), dev_recall9=float(totals[1] / totals[3]),
                dev_bce=float(totals[2] / totals[3]), dev_questions=int(totals[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2, help="Questions per process/GPU")
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--views-per-class", type=int, default=5)
    parser.add_argument("--view-batch", type=int, default=16, help="Dev scoring chunk size")
    parser.add_argument("--resume", action="store_true", help="Resume this run's last.pt at an epoch boundary")
    parser.add_argument("--max-train-questions", type=int, default=0, help="Smoke only; 0 means all")
    parser.add_argument("--max-dev-questions", type=int, default=0, help="Smoke only; 0 means all")
    parser.add_argument("--cpu", action="store_true", help="Small local tests, not full experiments")
    # Small dimensions are useful for numerical/DDP tests; production defaults match upstream.
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accum_steps, args.num_queries, args.views_per_class,
           args.view_batch, args.embedding_dim, args.output_dim, args.num_layers, args.num_heads) < 1:
        parser.error("Epochs, batch sizes and model dimensions must be positive")
    if args.workers < 0 or args.lr <= 0 or min(args.max_train_questions, args.max_dev_questions) < 0:
        parser.error("Invalid worker count, learning rate or smoke limits")
    if args.embedding_dim % args.num_heads:
        parser.error("embedding-dim must be divisible by num-heads")
    rank, world, device = distributed_device(args.cpu)
    seed_everything(args.seed + rank)
    cache, run = args.cache_dir.resolve(), args.run_dir.resolve()
    manifest = load_manifest(cache, "train")  # Never open val annotations/features here.
    train_rows, dev_rows = scene_split(manifest["rows"], args.dev_fraction, args.seed)
    if args.max_train_questions:
        train_rows = train_rows[:args.max_train_questions]
    if args.max_dev_questions:
        dev_rows = dev_rows[:args.max_dev_questions]
    if len(train_rows) < world:
        raise ValueError("Need at least one training question per process")
    model_config = dict(input_channel=manifest["input_channel"], embedding_dim=args.embedding_dim,
                        flat_out_size=args.output_dim, num_layers=args.num_layers,
                        num_heads=args.num_heads, num_queries=args.num_queries)
    config = {**vars(args), "cache_dir": str(cache), "run_dir": str(run), "world_size": world,
              "effective_batch_questions": args.batch_size * world * args.accum_steps,
              "train_manifest_sha256": sha256(cache / "train.json"), "cache_id": manifest["cache_id"],
              "model_config": model_config, "selection_metric": "dev_ap", "question_balanced_bce": True}
    # Epoch budget may be extended on resume; data/model/batch/seed must match.
    signature = {k: v for k, v in config.items() if k not in ("epochs", "resume")}
    if rank == 0:
        if args.resume:
            if read_json(run / "config.json")["signature"] != signature:
                raise ValueError("Resume config/data differs from the original run")
        else:
            run.mkdir(parents=True, exist_ok=False)
            revision = code_record(run)
            write_json(run / "config.json", dict(signature=signature, initial_epochs=args.epochs,
                                                git_commit=revision, device=str(device)))
            write_json(run / "split.json", {
                "source": "ScanQA train only; scene-disjoint internal dev",
                "train_scenes": sorted({r["scene_id"] for r in train_rows}),
                "dev_scenes": sorted({r["scene_id"] for r in dev_rows}),
                "train_ids": [r["question_id"] for r in train_rows],
                "dev_ids": [r["question_id"] for r in dev_rows]})
    barrier()
    model = QueryViewSelector(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start_epoch, best_ap = 0, -1.0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_ap = checkpoint["epoch"], checkpoint["best_ap"]
    trained = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model
    if checkpoint is not None:
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    dataset = QuestionDataset(train_rows, cache, args.seed, args.views_per_class)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_questions, pin_memory=device.type == "cuda")
    if rank == 0:
        print(f"R2: {sum(p.numel() for p in model.parameters()):,} trainable parameters; "
              f"world={world}; effective batch={config['effective_batch_questions']} questions; "
              f"train/dev={len(train_rows)}/{len(dev_rows)}; no 7B model loaded", flush=True)
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        trained.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for step, batch in enumerate(loader):
            boundary = (step + 1) % args.accum_steps == 0 or step + 1 == len(loader)
            window_start = (step // args.accum_steps) * args.accum_steps * args.batch_size
            local_questions = min(args.batch_size * args.accum_steps, len(sampler) - window_start)
            context = trained.no_sync() if world > 1 and not boundary else nullcontext()
            with context:
                scores = trained(**model_inputs(batch, device))
                losses = binary_losses(scores, batch["labels"].to(device))
                per_question = torch.stack([x.mean() for x in losses.split(batch["counts"])])
                # Same local count on every DDP rank, including the final partial window.
                (per_question.sum() / local_questions).backward()
            totals += torch.tensor([float(per_question.detach().sum()), len(batch["counts"])], device=device)
            if boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if rank == 0 and (step % 50 == 0 or step + 1 == len(loader)):
                print(f"epoch={epoch+1} step={step+1}/{len(loader)} bce={float(per_question.mean()):.5f}", flush=True)
        if world > 1:
            dist.all_reduce(totals)
        metrics = validate(model, dev_rows, cache, device, rank, world, args.view_batch)
        metrics.update(epoch=epoch + 1, train_bce=float(totals[0] / totals[1]),
                       seconds=time.perf_counter() - started,
                       sampled_train_questions=int(totals[1]))
        improved = metrics["dev_ap"] > best_ap
        best_ap = max(best_ap, metrics["dev_ap"])
        rng = {"cpu": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        rngs = [rng]
        if world > 1:
            rngs = [None] * world
            dist.all_gather_object(rngs, rng)
        if rank == 0:
            state = dict(format_version=1, architecture="r2_query_pool", model_config=model_config,
                         model=model.state_dict(), epoch=epoch + 1, best_ap=best_ap,
                         cache_id=manifest["cache_id"], train_manifest_sha256=config["train_manifest_sha256"],
                         smoke=bool(args.max_train_questions or args.max_dev_questions))
            if improved:
                atomic_save(state, run / "best.pt")
            atomic_save({**state, "optimizer": optimizer.state_dict(), "rng_by_rank": rngs}, run / "last.pt")
            with (run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Small runtime helpers shared by feature extraction, DDP training and ranking."""

from datetime import timedelta
import os
from pathlib import Path
import random
import subprocess

import torch
import torch.distributed as dist


def distributed_device(cpu=False):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else f"cuda:{local_rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2))
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def code_record(folder):
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True)
    (Path(folder) / "code.diff").write_text(diff.stdout)
    sources = Path(folder) / "source"
    sources.mkdir(exist_ok=True)
    for name in ("r2_model.py", "r2_data.py", "r2_runtime.py", "ViewSelector.py",
                 "prepare_r2_features.py", "train_r2.py", "rank_r2.py"):
        (sources / name).write_bytes((repo / "cdviews" / name).read_bytes())
    return revision.stdout.strip()

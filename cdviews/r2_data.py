"""Frozen feature loading, scene split and question-balanced selector batches."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_manifest(cache, split):
    cache = Path(cache)
    manifest = read_json(cache / f"{split}.json")
    if manifest["split"] != split or manifest["cache_id"] != fingerprint(read_json(cache / "config.json")):
        raise ValueError("Feature manifest/config mismatch; prepare a consistent cache first")
    return manifest


def scene_split(rows, fraction=0.1, seed=42):
    scenes = sorted({row["scene_id"] for row in rows})
    if len(scenes) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two training scenes and dev fraction in (0,1)")
    random.Random(seed).shuffle(scenes)
    count = min(len(scenes) - 1, max(1, round(len(scenes) * fraction)))
    dev_scenes = set(scenes[:count])
    train = [row for row in rows if row["scene_id"] not in dev_scenes]
    dev = [row for row in rows if row["scene_id"] in dev_scenes]
    return train, dev


class FeatureStore:
    def __init__(self, root, split):
        self.root, self.split = Path(root), split

    @lru_cache(maxsize=32)
    def image(self, scene, name):
        return torch.load(self.root / "images" / scene / (name + ".pt"),
                          map_location="cpu", weights_only=True)

    def text(self, qid):
        return torch.load(self.root / "questions" / self.split / (str(qid) + ".pt"),
                          map_location="cpu", weights_only=True)

    def example(self, row, names, labels):
        return ([self.image(row["scene_id"], name) for name in names],
                self.text(row["question_id"]), torch.tensor(labels, dtype=torch.float32))


class QuestionDataset(Dataset):
    def __init__(self, rows, cache, seed=42, per_class=5):
        self.rows, self.store = rows, FeatureStore(cache, "train")
        self.seed, self.per_class, self.epoch = seed, per_class, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        key = f"{self.seed}:{self.epoch}:{row['question_id']}"
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
        n = min(self.per_class, len(row["positive"]), len(row["negative"]))
        positive = rng.sample(row["positive"], n)
        negative = rng.sample(row["negative"], n)
        return self.store.example(row, positive + negative, [1.] * n + [0.] * n)


def collate_questions(batch):
    images, texts, labels, counts = [], [], [], []
    for visual, question, target in batch:
        images.extend(visual)
        texts.extend([question] * len(visual))
        labels.append(target)
        counts.append(len(visual))
    image_lengths = torch.tensor([len(x) for x in images])
    text_lengths = torch.tensor([len(x) for x in texts])
    images = pad_sequence(images, batch_first=True)
    text = pad_sequence(texts, batch_first=True)
    return dict(images=images, text=text,
                image_padding=torch.arange(images.shape[1])[None] >= image_lengths[:, None],
                text_padding=torch.arange(text.shape[1])[None] >= text_lengths[:, None],
                labels=torch.cat(labels), counts=counts)


def model_inputs(batch, device):
    return {name: batch[name].to(device=device, dtype=torch.float32 if name in ("images", "text") else None)
            for name in ("images", "text", "image_padding", "text_padding")}


@torch.no_grad()
def score_question(model, store, row, names, device, batch_size=16):
    scores = []
    text = store.text(row["question_id"])
    for start in range(0, len(names), batch_size):
        chunk = names[start:start + batch_size]
        images = [store.image(row["scene_id"], name) for name in chunk]
        batch = collate_questions([(images, text, torch.zeros(len(chunk)))])
        scores.append(model(**model_inputs(batch, device)).cpu())
    return torch.cat(scores)


def average_precision(scores, labels):
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = labels[order].float()
    precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / ordered.sum())

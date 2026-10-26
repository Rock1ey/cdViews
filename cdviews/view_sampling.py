"""Reproducible uniform sampling from the same candidate pool as cdViews."""

import hashlib
import random


def question_seed(seed, split, question_id, stream):
    """Separate view sampling and decoding; independent of question/shard order."""
    key = f"{seed}:{split}:{question_id}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def uniform_views(candidates, num_views, seed, split, question_id):
    # Sorting removes the selector's ranking signal. Keep the random draw order
    # as the LVLM input order; do not sort sampled frames back by relevance.
    population = sorted(candidates)
    rng = random.Random(question_seed(seed, split, question_id, "views"))
    return rng.sample(population, min(num_views, len(population)))

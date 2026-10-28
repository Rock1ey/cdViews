"""Training-free, rank-based soft view selection (experimental baseline).

Relevance comes from the original question-specific cdViews ranking, not from
BLIP question retrieval or calibrated selector scores. Image features must be
L2-normalized and ordered exactly like candidates; distances use the same order.
"""

import numpy as np


def soft_select_views(candidates, image_features, pose_distances, num_views,
                      penalty=0.5, rank_scale=10.0, pose_scale=0.5):
    """Greedily balance original rank against maximum selected-view redundancy.

    r(i) = 1 / (1 + zero_based_rank(i) / rank_scale)
    overlap(i,j) = max(0, cosine(i,j)) * (0.5 + 0.5 * exp(-distance(i,j)/pose_scale))
    utility(i | S) = r(i) - penalty * max_{j in S} overlap(i,j)

    Distant duplicate content still receives a penalty; proximity strengthens
    it, but cannot by itself suppress visually complementary views. No candidate
    is hard-deleted. The scales are initial hypotheses, not tuned parameters.
    """
    if num_views < 1 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("num_views must be positive and penalty finite/nonnegative")
    if not np.isfinite(rank_scale) or not np.isfinite(pose_scale) or min(rank_scale, pose_scale) <= 0:
        raise ValueError("rank_scale and pose_scale must be finite and positive")
    relevance = 1.0 / (1.0 + np.arange(len(candidates)) / rank_scale)
    redundancy = np.zeros(len(candidates))
    available = np.ones(len(candidates), dtype=bool)
    selected, trace = [], []

    for _ in range(min(num_views, len(candidates))):
        utility = relevance - penalty * redundancy
        # np.argmax resolves ties by the original cdViews rank.
        index = int(np.argmax(np.where(available, utility, -np.inf)))
        selected.append(candidates[index])
        trace.append({"image": candidates[index], "original_rank": index + 1,
                      "rank_relevance": float(relevance[index]),
                      "max_redundancy": float(redundancy[index]),
                      "selection_score": float(utility[index])})
        available[index] = False
        visual_similarity = np.clip(image_features @ image_features[index], 0.0, 1.0)
        proximity = np.exp(-pose_distances[:, index] / pose_scale)
        overlap = visual_similarity * (0.5 + 0.5 * proximity)
        redundancy = np.maximum(redundancy, overlap)

    return selected, trace

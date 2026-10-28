"""Training-free, rank-based soft view selection (experimental baseline).

Relevance comes from the original question-specific cdViews ranking, not from
BLIP question retrieval or calibrated selector scores. Image features must be
L2-normalized and ordered exactly like candidates; distances use the same order.
"""

import numpy as np


def question_coverage_weights(image_features, question_feature, rank_scale=10.0,
                              temperature=0.07, text_weight=1.0):
    """Weight candidate content by original rank and BLIP question similarity.

    These are normalized coverage weights, not answer correctness probabilities.
    text_weight=0 is the rank-only coverage ablation.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("coverage temperature must be finite and positive")
    if not np.isfinite(text_weight) or not 0 <= text_weight <= 1:
        raise ValueError("coverage text_weight must be in [0, 1]")
    prior = 1.0 / (1.0 + np.arange(len(image_features)) / rank_scale)
    similarity = image_features @ question_feature
    logits = np.log(prior) + text_weight * similarity / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum(), similarity


def soft_select_views(candidates, image_features, pose_distances, num_views,
                      penalty=0.5, rank_scale=10.0, pose_scale=0.5,
                      coverage_kernel=None, coverage_weights=None, coverage_weight=0.0):
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
    if not np.isfinite(coverage_weight) or coverage_weight < 0:
        raise ValueError("coverage_weight must be finite and nonnegative")
    use_coverage = coverage_weight > 0
    if use_coverage and (coverage_kernel is None or coverage_weights is None):
        raise ValueError("Positive coverage_weight requires a kernel and candidate weights")
    relevance = 1.0 / (1.0 + np.arange(len(candidates)) / rank_scale)
    redundancy = np.zeros(len(candidates))
    available = np.ones(len(candidates), dtype=bool)
    selected, trace = [], []
    covered = np.zeros(len(candidates))

    for _ in range(min(num_views, len(candidates))):
        utility = relevance - penalty * redundancy
        gains = np.zeros(len(candidates))
        if use_coverage:
            # kernel[u,i]: how well view i represents candidate content u.
            # F(S) = sum_u w[u] * max_{j in S} kernel[u,j]. Gains shrink as
            # selected views cover that content. No labels or answer feedback.
            gains = coverage_weights @ np.maximum(coverage_kernel - covered[:, None], 0.0)
            if selected:
                utility = utility + coverage_weight * gains
        # Keep the original top-ranked view as the common anchor in both modes.
        # np.argmax resolves ties by the original cdViews rank.
        index = int(np.argmax(np.where(available, utility, -np.inf)))
        selected.append(candidates[index])
        trace.append({"image": candidates[index], "original_rank": index + 1,
                      "rank_relevance": float(relevance[index]),
                      "max_redundancy": float(redundancy[index]),
                      "selection_score": float(utility[index])})
        if use_coverage:
            covered = np.maximum(covered, coverage_kernel[:, index])
            trace[-1].update({"coverage_gain": float(gains[index]),
                              "weighted_coverage": float(coverage_weights @ covered),
                              "coverage_bonus": float(coverage_weight * gains[index]) if len(selected) > 1 else 0.0,
                              "anchor": len(selected) == 1})
        available[index] = False
        visual_similarity = np.clip(image_features @ image_features[index], 0.0, 1.0)
        proximity = np.exp(-pose_distances[:, index] / pose_scale)
        overlap = visual_similarity * (0.5 + 0.5 * proximity)
        redundancy = np.maximum(redundancy, overlap)

    return selected, trace

"""Numerical checks of the experimental selection objective; no GPU needed."""

import unittest

import numpy as np

from cdviews.soft_selection import soft_select_views, question_coverage_weights


class SoftSelectionTests(unittest.TestCase):
    def setUp(self):
        self.names = ["anchor.jpg", "duplicate.jpg", "detail.jpg", "other.jpg"]
        self.features = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.distances = np.zeros((4, 4))

    def test_nearby_complement_survives_and_duplicate_is_delayed(self):
        selected, trace = soft_select_views(self.names, self.features, self.distances, 2)
        self.assertEqual(selected, ["anchor.jpg", "detail.jpg"])
        self.assertEqual(trace[1]["original_rank"], 3)
        self.assertEqual(trace[1]["max_redundancy"], 0)

    def test_zero_penalty_recovers_original_top_k(self):
        selected, _ = soft_select_views(self.names, self.features, self.distances, 3, penalty=0)
        self.assertEqual(selected, self.names[:3])

    def test_duplicate_content_penalized_even_at_distant_pose(self):
        self.distances[0, 1] = self.distances[1, 0] = 100
        selected, trace = soft_select_views(self.names[:3], self.features[:3], self.distances[:3, :3], 3)
        self.assertEqual(selected, ["anchor.jpg", "detail.jpg", "duplicate.jpg"])
        self.assertAlmostEqual(trace[2]["max_redundancy"], 0.5)

    def test_pose_changes_order_when_visual_similarity_is_equal(self):
        features = np.ones((3, 1))
        distances = np.array([[0., 0., 100.], [0., 0., 100.], [100., 100., 0.]])
        selected, _ = soft_select_views(self.names[:3], features, distances, 2)
        self.assertEqual(selected, ["anchor.jpg", "detail.jpg"])
        selected, _ = soft_select_views(self.names[:3], features, np.zeros((3, 3)), 2)
        self.assertEqual(selected, self.names[:2])

    def test_soft_selection_never_hard_deletes_or_duplicates(self):
        selected, _ = soft_select_views(self.names, np.ones((4, 1)), self.distances, 9)
        self.assertEqual(selected, self.names)

    def test_invalid_scales_rejected(self):
        for kwargs in ({"penalty": -1}, {"pose_scale": 0}, {"rank_scale": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                soft_select_views(self.names, self.features, self.distances, 2, **kwargs)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.names = ["anchor", "side_a", "side_b"]
        self.features = np.eye(3)
        self.kernel = np.eye(3)
        self.distances = np.zeros((3, 3))

    def select(self, question, **kwargs):
        weights, _ = question_coverage_weights(self.features, question)
        return soft_select_views(self.names, self.features, self.distances, 3,
                                 coverage_kernel=self.kernel, coverage_weights=weights,
                                 coverage_weight=0.5, **kwargs)

    def test_question_changes_complement_but_keeps_anchor(self):
        first, _ = self.select(np.array([0., 1., 0.]))
        second, _ = self.select(np.array([0., 0., 1.]))
        self.assertEqual(first, ["anchor", "side_a", "side_b"])
        self.assertEqual(second, ["anchor", "side_b", "side_a"])

    def test_covered_cluster_has_no_further_gain(self):
        names = ["anchor", "cluster_a", "duplicate_a", "cluster_b"]
        features = np.array([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
        selected, trace = soft_select_views(
            names, features, np.zeros((4, 4)), 4, penalty=0,
            coverage_kernel=features @ features.T,
            coverage_weights=np.array([0.1, 0.3, 0.3, 0.3]), coverage_weight=1)
        self.assertEqual(selected, ["anchor", "cluster_a", "cluster_b", "duplicate_a"])
        self.assertAlmostEqual(trace[-1]["coverage_gain"], 0)
        self.assertAlmostEqual(trace[-1]["weighted_coverage"], 1)
        # Check marginal gains against the change in the actual set function.
        previous = 0
        for step in trace:
            self.assertAlmostEqual(step["coverage_gain"], step["weighted_coverage"] - previous)
            previous = step["weighted_coverage"]
        self.assertEqual(trace[0]["coverage_bonus"], 0)

    def test_zero_coverage_weight_exactly_recovers_v1_including_trace(self):
        base = soft_select_views(self.names, self.features, self.distances, 9)
        result = soft_select_views(self.names, self.features, self.distances, 9,
                                   coverage_kernel=self.kernel, coverage_weights=np.array([0., 0., 1.]),
                                   coverage_weight=0)
        self.assertEqual(base, result)

    def test_rank_only_ablation_ignores_question_and_weights_are_normalized(self):
        first, _ = question_coverage_weights(self.features, np.array([0., 1., 0.]), text_weight=0)
        second, _ = question_coverage_weights(self.features, np.array([0., 0., 1.]), text_weight=0)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(first.sum(), 1)
        self.assertGreater(first[0], first[1])

    def test_weight_stability_and_small_pool(self):
        weights, _ = question_coverage_weights(self.features, np.array([1., 0., 0.]), temperature=1e-4)
        self.assertTrue(np.isfinite(weights).all())
        selected, trace = soft_select_views(["only"], np.ones((1, 1)), np.zeros((1, 1)), 9,
                                            coverage_kernel=np.ones((1, 1)), coverage_weights=np.ones(1),
                                            coverage_weight=0.5)
        self.assertEqual(selected, ["only"])
        self.assertEqual(trace[0]["weighted_coverage"], 1)


if __name__ == "__main__":
    unittest.main()

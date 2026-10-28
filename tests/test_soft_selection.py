"""Numerical checks of the experimental selection objective; no GPU needed."""

import unittest

import numpy as np

from cdviews.soft_selection import soft_select_views


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


if __name__ == "__main__":
    unittest.main()

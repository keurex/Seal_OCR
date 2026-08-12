"""合成印章方向分布和大角度旋转范围的回归测试。"""

from __future__ import annotations

import random
import unittest

from synthesis.generate import (
    DEFAULT_LARGE_ROTATION_PROBABILITY,
    LARGE_ROTATION_MAX_DEGREES,
    LARGE_ROTATION_MIN_DEGREES,
    DEFAULT_STYLE,
    Stamp,
)


class RotationStrategyTest(unittest.TestCase):
    def test_default_keeps_25_percent_large_rotations(self) -> None:
        self.assertEqual(DEFAULT_LARGE_ROTATION_PROBABILITY, 0.25)
        self.assertEqual(
            DEFAULT_STYLE["large_rotation_probability"],
            DEFAULT_LARGE_ROTATION_PROBABILITY,
        )

    def test_large_rotation_is_never_less_than_45_degrees(self) -> None:
        stamp = Stamp()
        stamp.large_rotation_probability = 1.0
        angles = []
        for seed in range(256):
            random.seed(seed)
            angles.append(stamp._choose_rotation())

        self.assertTrue(
            all(
                LARGE_ROTATION_MIN_DEGREES <= abs(angle)
                <= LARGE_ROTATION_MAX_DEGREES
                for angle in angles
            )
        )
        self.assertTrue(any(angle < 0 for angle in angles))
        self.assertTrue(any(angle > 0 for angle in angles))

    def test_rotation_bounds_are_configurable(self) -> None:
        stamp = Stamp()
        stamp.large_rotation_probability = 1.0
        stamp.large_rotation_min_degrees = 70.0
        stamp.large_rotation_max_degrees = 80.0
        for seed in range(64):
            random.seed(seed)
            angle = abs(stamp._choose_rotation())
            self.assertLessEqual(70.0, angle)
            self.assertLessEqual(angle, 80.0)

    def test_upright_branch_keeps_existing_small_tilt(self) -> None:
        stamp = Stamp()
        stamp.large_rotation_probability = 0.0
        for shape, layout, maximum in (
            ("circle", "standard", 18.0),
            ("ellipse", "oval_service", 25.0),
            ("rectangle", "standard", 7.0),
        ):
            stamp.shape = shape
            stamp.layout = layout
            for seed in range(64):
                random.seed(seed)
                with self.subTest(shape=shape, layout=layout, seed=seed):
                    self.assertLessEqual(abs(stamp._choose_rotation()), maximum)


if __name__ == "__main__":
    unittest.main()

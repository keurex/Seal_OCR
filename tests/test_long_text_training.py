"""长名称、视觉位置编码与语言先验抑制的回归测试。"""

from __future__ import annotations

import unittest
from math import cos, pi, sin

from synthesis.generate import _equal_ellipse_arc_angles
from seal_ocr.initialize import _resize_vision_position_embeddings
from train import (
    apply_decoder_context_dropout,
    compute_length_aware_metrics,
)

try:
    import torch
except ModuleNotFoundError:  # 开发机可只执行不依赖 PyTorch 的指标测试
    torch = None


def _sample_error_rate(*, predictions, references):
    return sum(
        predicted != expected
        for predicted, expected in zip(predictions, references)
    ) / max(len(references), 1)


class LengthAwareMetricsTest(unittest.TestCase):
    def test_each_length_bucket_has_equal_selection_weight(self) -> None:
        references = ["甲" * 10, "乙" * 14, "丙" * 18, "丁" * 20]
        predictions = [references[0], "错", references[2], "错"]

        metrics = compute_length_aware_metrics(
            predictions,
            references,
            cer_function=_sample_error_rate,
        )

        self.assertEqual(metrics["exact_match_len_le_12"], 1.0)
        self.assertEqual(metrics["exact_match_len_13_16"], 0.0)
        self.assertEqual(metrics["exact_match_len_17_19"], 1.0)
        self.assertEqual(metrics["exact_match_len_ge_20"], 0.0)
        self.assertEqual(metrics["length_balanced_exact_match"], 0.5)
        self.assertEqual(metrics["long_exact_match"], 0.5)
        self.assertEqual(metrics["long_samples"], 2)

    def test_ellipse_characters_are_evenly_spaced_after_compression(self) -> None:
        ellipse_ratio = 0.72
        angles = _equal_ellipse_arc_angles(300, 18, ellipse_ratio)
        points = [
            (
                -sin(angle * pi / 180),
                -cos(angle * pi / 180) * ellipse_ratio,
            )
            for angle in angles
        ]
        distances = [
            ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
            ** 0.5
            for left, right in zip(points, points[1:])
        ]

        self.assertLess(max(distances) / min(distances), 1.08)


@unittest.skipIf(torch is None, "需要 PyTorch")
class VisionAndDecoderRegularizationTest(unittest.TestCase):
    def test_deit_two_prefix_tokens_are_not_interpolated(self) -> None:
        source = torch.arange(578 * 4, dtype=torch.float32).reshape(1, 578, 4)
        target = torch.empty((1, 1026, 4), dtype=torch.float32)

        resized = _resize_vision_position_embeddings(torch, source, target)

        self.assertEqual(tuple(resized.shape), tuple(target.shape))
        torch.testing.assert_close(resized[:, :2], source[:, :2])
        self.assertTrue(torch.isfinite(resized).all())

    def test_context_dropout_preserves_special_tokens(self) -> None:
        decoder_input_ids = torch.tensor([[0, 0, 5, 6, 2, 1]])

        masked = apply_decoder_context_dropout(
            torch,
            decoder_input_ids,
            probability=1.0,
            replacement_token_id=4,
            protected_token_ids=(0, 1, 2, 4),
        )

        self.assertEqual(masked.tolist(), [[0, 0, 4, 4, 2, 1]])
        self.assertEqual(decoder_input_ids.tolist(), [[0, 0, 5, 6, 2, 1]])


if __name__ == "__main__":
    unittest.main()

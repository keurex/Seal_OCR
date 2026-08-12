"""预训练阶段空间标注引导的笔画级退化回归测试。"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image, ImageDraw, ImageFilter
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFilter = None

from train import STAGE_DEFAULTS

if Image is not None and np is not None:
    from seal_ocr.degradation import DegradationConfig, apply_degradation


@unittest.skipIf(Image is None or np is None, "Pillow 或 NumPy 未安装")
class DegradationTest(unittest.TestCase):
    def _sample(self):
        width = height = 192
        image = Image.new("RGB", (width, height), "white")
        text_mask = Image.new("L", (width, height), 0)
        stamp_mask = Image.new("L", (width, height), 0)
        text_draw = ImageDraw.Draw(text_mask)
        stamp_draw = ImageDraw.Draw(stamp_mask)
        stamp_draw.ellipse((30, 30, 162, 162), outline=255, width=6)
        image_draw = ImageDraw.Draw(image)
        for index in range(12):
            x0 = 38 + index * 10
            y0 = 45 + (index % 3) * 3
            box = (x0, y0, x0 + 6, y0 + 28)
            text_draw.rectangle(box, fill=255)
            stamp_draw.rectangle(box, fill=255)
            image_draw.rectangle(box, fill=(196, 25, 30))
        annotation = Image.merge(
            "RGBA",
            (
                text_mask,
                stamp_mask,
                text_mask.filter(ImageFilter.GaussianBlur(radius=4)),
                stamp_mask.filter(ImageFilter.GaussianBlur(radius=5)),
            ),
        )
        return image, annotation

    def test_disabled_configuration_keeps_pixels_unchanged(self) -> None:
        image, annotation = self._sample()
        result, stats = apply_degradation(
            image,
            annotation,
            config=DegradationConfig(enabled=False),
            rng=np.random.default_rng(11),
        )
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertFalse(stats.applied)

    def test_dissolution_is_granular_and_preserves_pixel_contrast(self) -> None:
        image, annotation = self._sample()
        annotation_before = annotation.tobytes()
        config = DegradationConfig(
            probability=1.0,
            text_dissolution_probability=1.0,
            background_clutter_probability=0.0,
            foreground_stroke_probability=0.0,
            max_text_dissolution_ratio=0.70,
            min_text_residual_ratio=0.25,
        )
        result, stats = apply_degradation(
            image,
            annotation,
            label_length=12,
            config=config,
            rng=np.random.default_rng(20260811),
        )
        original_array = np.asarray(image, dtype=np.float32)
        result_array = np.asarray(result, dtype=np.float32)
        text_mask = np.asarray(annotation)[:, :, 0] >= 128
        changed = text_mask & np.any(result_array != original_array, axis=2)

        self.assertTrue(stats.text_dissolution_applied)
        self.assertGreater(stats.text_dissolution_ratio, 0.20)
        self.assertLessEqual(stats.text_dissolution_ratio, 0.70)
        self.assertGreaterEqual(stats.minimum_text_residual_ratio, 0.25)
        self.assertTrue(changed.any())
        self.assertTrue(
            np.array_equal(
                result_array[~text_mask],
                original_array[~text_mask],
            )
        )

        paper = np.full_like(original_array[changed], 255.0)
        original_contrast = np.linalg.norm(
            paper - original_array[changed],
            axis=1,
        )
        remaining_contrast = np.linalg.norm(
            paper - result_array[changed],
            axis=1,
        )
        residual_ratio = remaining_contrast / original_contrast
        # uint8 舍入允许极小误差，但任何被消解像素都不能成为纯纸面空洞。
        self.assertGreaterEqual(float(residual_ratio.min()), 0.245)
        self.assertEqual(annotation.tobytes(), annotation_before)

    def test_long_name_keeps_more_residual_and_lower_coverage(self) -> None:
        image, annotation = self._sample()
        _, stats = apply_degradation(
            image,
            annotation,
            label_length=20,
            config=DegradationConfig(
                probability=1.0,
                text_dissolution_probability=1.0,
                background_clutter_probability=0.0,
                foreground_stroke_probability=0.0,
                max_text_dissolution_ratio=0.70,
                min_text_residual_ratio=0.25,
            ),
            rng=np.random.default_rng(72),
        )
        self.assertTrue(stats.text_dissolution_applied)
        self.assertLessEqual(stats.text_dissolution_ratio, 0.70 * 0.70)
        self.assertGreaterEqual(stats.minimum_text_residual_ratio, 0.30)

    def test_foreground_occlusion_is_only_thin_strokes_and_dots(self) -> None:
        image, annotation = self._sample()
        result, stats = apply_degradation(
            image,
            annotation,
            config=DegradationConfig(
                probability=1.0,
                text_dissolution_probability=0.0,
                background_clutter_probability=0.0,
                foreground_stroke_probability=1.0,
                max_foreground_text_overlap_ratio=0.06,
            ),
            rng=np.random.default_rng(314159),
        )
        self.assertTrue(stats.foreground_stroke_applied)
        self.assertLessEqual(stats.foreground_text_overlap_ratio, 0.06)
        self.assertNotEqual(result.tobytes(), image.tobytes())

    def test_dense_document_clutter_does_not_replace_stamp_pixels(self) -> None:
        image, annotation = self._sample()
        result, stats = apply_degradation(
            image,
            annotation,
            config=DegradationConfig(
                probability=1.0,
                text_dissolution_probability=0.0,
                background_clutter_probability=1.0,
                foreground_stroke_probability=0.0,
            ),
            rng=np.random.default_rng(271828),
        )
        self.assertTrue(stats.background_clutter_applied)
        self.assertGreater(stats.clutter_coverage, 0.0)
        self.assertLess(
            np.asarray(result, dtype=np.float32).mean(),
            np.asarray(image, dtype=np.float32).mean(),
        )

    def test_configuration_rejects_feature_erasing_values(self) -> None:
        with self.assertRaises(ValueError):
            DegradationConfig(min_text_residual_ratio=0.10)
        with self.assertRaises(ValueError):
            DegradationConfig(max_foreground_text_overlap_ratio=0.20)


class DegradationStageDefaultsTest(unittest.TestCase):
    def test_only_pretrain_enables_targeted_degradation(self) -> None:
        self.assertTrue(STAGE_DEFAULTS["pretrain"]["degradation_enabled"])
        self.assertFalse(STAGE_DEFAULTS["finetune"]["degradation_enabled"])

    def test_default_guarantees_pixel_level_residual(self) -> None:
        defaults = STAGE_DEFAULTS["pretrain"]
        self.assertEqual(defaults["degradation_probability"], 0.60)
        self.assertEqual(defaults["min_text_residual_ratio"], 0.25)
        self.assertLessEqual(
            defaults["max_foreground_text_overlap_ratio"],
            0.06,
        )


if __name__ == "__main__":
    unittest.main()

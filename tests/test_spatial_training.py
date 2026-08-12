"""空间辅助监督的数据配对、尺寸对齐和损失回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from seal_ocr.data import discover_samples
from train import STAGE_DEFAULTS

if Image is not None:
    from seal_ocr.image import (
        prepare_spatial_annotation_bundle_for_processor,
        prepare_spatial_annotation_for_processor,
    )
    from seal_ocr.spatial_annotations import SpatialAnnotationBundle

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from seal_ocr.dataset import identity_image_transform
except ModuleNotFoundError:
    identity_image_transform = None

if torch is not None:
    from seal_ocr.length_control import scale_gradient
    from seal_ocr.spatial_control import (
        SpatialAuxiliaryHead,
        compute_spatial_objective,
    )


@unittest.skipIf(Image is None, "Pillow 未安装")
class SpatialDataDiscoveryTest(unittest.TestCase):
    def _write_sample(self, root: Path, with_annotation: bool) -> Path:
        image_root = root / "synthetic"
        shard = image_root / "00000"
        shard.mkdir(parents=True)
        Image.new("RGB", (16, 12), "white").save(shard / "7.jpg")
        (shard / "7.txt").write_text("测试科技有限公司", encoding="utf-8")
        if with_annotation:
            spatial_shard = root / "synthetic_spatial" / "00000"
            spatial_shard.mkdir(parents=True)
            Image.new("RGBA", (16, 12), (0, 0, 0, 0)).save(
                spatial_shard / "7.png"
            )
        return image_root

    def test_auto_discovers_sibling_spatial_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_root = self._write_sample(Path(temporary_dir), True)
            samples, issues = discover_samples(
                [str(image_root)],
                auto_spatial_annotations=True,
                require_spatial_annotations=True,
            )
            self.assertFalse(issues)
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0].spatial_annotation_path.endswith("7.png"))

    def test_missing_real_annotation_stays_none_without_issue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_root = self._write_sample(Path(temporary_dir), False)
            samples, issues = discover_samples(
                [str(image_root)],
                auto_spatial_annotations=True,
                require_spatial_annotations=False,
            )
            self.assertFalse(issues)
            self.assertEqual(len(samples), 1)
            self.assertIsNone(samples[0].spatial_annotation_path)

    def test_letterbox_uses_zero_padding_for_all_annotation_channels(self) -> None:
        class ImageProcessor:
            size = {"width": 8, "height": 8}

        class Processor:
            image_processor = ImageProcessor()

        channels = [Image.new("L", (8, 4), 255) for _ in range(4)]
        annotation = Image.merge("RGBA", channels)
        prepared = prepare_spatial_annotation_for_processor(
            annotation,
            processor=Processor(),
            resize_mode="letterbox",
        )
        self.assertEqual(prepared.size, (8, 8))
        for channel in prepared.split():
            self.assertEqual(channel.crop((0, 0, 8, 2)).getbbox(), None)
            self.assertEqual(channel.crop((0, 6, 8, 8)).getbbox(), None)
            self.assertIsNotNone(channel.crop((0, 2, 8, 6)).getbbox())

    def test_letterbox_keeps_v2_detail_channels_aligned(self) -> None:
        class ImageProcessor:
            size = {"width": 8, "height": 8}

        class Processor:
            image_processor = ImageProcessor()

        primary = Image.merge(
            "RGBA", [Image.new("L", (8, 4), 255) for _ in range(4)]
        )
        detail = Image.merge(
            "RGBA", [Image.new("L", (8, 4), 127) for _ in range(4)]
        )
        prepared = prepare_spatial_annotation_bundle_for_processor(
            SpatialAnnotationBundle(primary=primary, detail=detail),
            processor=Processor(),
            resize_mode="letterbox",
        )
        self.assertEqual(prepared.primary.size, (8, 8))
        self.assertEqual(prepared.detail.size, (8, 8))
        for image in (prepared.primary, prepared.detail):
            for channel in image.split():
                self.assertEqual(channel.crop((0, 0, 8, 2)).getbbox(), None)
                self.assertEqual(channel.crop((0, 6, 8, 8)).getbbox(), None)


class SpatialStageDefaultsTest(unittest.TestCase):
    def test_public_flow_has_two_fixed_stage_learning_rates(self) -> None:
        self.assertEqual(set(STAGE_DEFAULTS), {"pretrain", "finetune"})
        self.assertEqual(STAGE_DEFAULTS["pretrain"]["learning_rate"], 2e-4)
        self.assertEqual(STAGE_DEFAULTS["finetune"]["learning_rate"], 5e-6)

    def test_spatial_supervision_has_no_public_architecture_switch(self) -> None:
        self.assertTrue(
            all(
                "spatial_architecture_version" not in defaults
                for defaults in STAGE_DEFAULTS.values()
            )
        )

    def test_only_real_finetune_freezes_head_by_default(self) -> None:
        self.assertFalse(STAGE_DEFAULTS["pretrain"]["freeze_spatial_head"])
        self.assertTrue(STAGE_DEFAULTS["finetune"]["freeze_spatial_head"])

    def test_encoder_spatial_gradient_is_reduced_by_stage(self) -> None:
        self.assertEqual(
            STAGE_DEFAULTS["pretrain"]["spatial_encoder_gradient_scale"],
            0.10,
        )
        self.assertEqual(
            STAGE_DEFAULTS["finetune"]["spatial_encoder_gradient_scale"],
            0.03,
        )


@unittest.skipIf(identity_image_transform is None, "PyTorch 未安装")
class DatasetTransformCompatibilityTest(unittest.TestCase):
    def test_identity_transform_preserves_spatial_annotation(self) -> None:
        image = object()
        annotation = object()
        self.assertIs(identity_image_transform(image), image)
        self.assertEqual(
            identity_image_transform(
                image,
                label_length=8,
                spatial_annotation=annotation,
            ),
            (image, annotation),
        )


@unittest.skipIf(torch is None, "PyTorch 未安装")
class SpatialHeadTest(unittest.TestCase):
    def _targets(self, batch_size: int = 2):
        targets = torch.zeros(batch_size, 8, 4, 4)
        targets[:, 0, 1:3, 1:3] = 1.0
        targets[:, 1, :, :] = 1.0
        targets[:, 2, 1:3, 1:3] = 0.8
        targets[:, 3, :, :] = 0.6
        targets[:, 4, 1, 1] = 1.0
        targets[:, 4, 1, 2] = 1.0
        targets[:, 5, 1, 1] = 0.25
        targets[:, 5, 1, 2] = 0.75
        targets[:, 6, 1, 1] = 1.0
        targets[:, 7, 1, 2] = 1.0
        return targets

    def test_head_ignores_prefix_tokens_and_outputs_patch_grid(self) -> None:
        head = SpatialAuxiliaryHead(
            encoder_hidden_size=8,
            head_hidden_size=6,
            grid_size=(4, 4),
        )
        encoder_hidden = torch.randn(2, 18, 8)
        self.assertEqual(tuple(head(encoder_hidden).shape), (2, 8, 4, 4))

    def test_matching_logits_have_lower_objective(self) -> None:
        targets = self._targets()
        matching_logits = torch.logit(targets.clamp(0.01, 0.99))
        inverted_logits = -matching_logits
        matching_loss, metrics = compute_spatial_objective(
            matching_logits,
            targets,
        )
        inverted_loss, _ = compute_spatial_objective(
            inverted_logits,
            targets,
        )
        self.assertLess(matching_loss.item(), inverted_loss.item())
        self.assertGreater(metrics["spatial_text_mask_dice"].item(), 0.99)

    def test_frozen_head_still_passes_scaled_gradient_to_encoder(self) -> None:
        head = SpatialAuxiliaryHead(
            encoder_hidden_size=8,
            head_hidden_size=6,
            grid_size=(4, 4),
        )
        for parameter in head.parameters():
            parameter.requires_grad = False
        encoder_hidden = torch.randn(2, 18, 8, requires_grad=True)
        logits = head(scale_gradient(encoder_hidden, 0.1))
        loss, _ = compute_spatial_objective(logits, self._targets())
        loss.backward()
        self.assertIsNotNone(encoder_hidden.grad)
        self.assertGreater(encoder_hidden.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_head_outputs_character_order_channels_and_metrics(self) -> None:
        head = SpatialAuxiliaryHead(
            encoder_hidden_size=8,
            head_hidden_size=6,
            grid_size=(4, 4),
        )
        encoder_hidden = torch.randn(2, 18, 8)
        logits = head(encoder_hidden)
        self.assertEqual(tuple(logits.shape), (2, 8, 4, 4))
        targets = self._targets()
        matching_logits = torch.logit(targets.clamp(0.01, 0.99))
        loss, metrics = compute_spatial_objective(
            matching_logits,
            targets,
        )
        inverted_loss, _ = compute_spatial_objective(
            -matching_logits,
            targets,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), inverted_loss.item())
        self.assertIn("spatial_character_reading_progress_mae", metrics)
        self.assertIn(
            "spatial_character_center_mass_relative_error",
            metrics,
        )
        self.assertIn("spatial_first_character_distance", metrics)
        self.assertLess(
            metrics["spatial_first_character_distance"].item(),
            0.01,
        )

if __name__ == "__main__":
    unittest.main()

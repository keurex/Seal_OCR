"""合成印章空间标注的几何对齐回归测试。"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError:  # 开发机可能只安装训练依赖。
    Image = None

if Image is not None:
    from synthesis.generate import (
        LARGE_ROTATION_MAX_DEGREES,
        LARGE_ROTATION_MIN_DEGREES,
        Stamp,
        build_parser,
    )


@unittest.skipIf(Image is None, "Pillow 未安装")
class SpatialAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        candidates = (
            Path("synthesis/fonts"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        )
        for candidate in candidates:
            if candidate.is_file():
                cls.font_path = candidate
                return
            if candidate.is_dir():
                font = next(
                    (
                        path
                        for path in candidate.iterdir()
                        if path.suffix.lower() in {".ttf", ".ttc", ".otf"}
                    ),
                    None,
                )
                if font is not None:
                    cls.font_path = font
                    return
        raise unittest.SkipTest("未找到可用于空间标注测试的中文字体")

    def _build_stamp(
        self,
        layout: str,
        shape: str,
        seed: int,
        large_rotation_probability: float = 0.0,
    ) -> Stamp:
        random.seed(seed)
        stamp = Stamp()
        stamp.layout = layout
        stamp.shape = shape
        stamp.ellipse_ratio = 0.75
        stamp.company_name = "上海测试科技有限公司"
        stamp.font_path = str(self.font_path)
        stamp.auxiliary_text = "业务专用章"
        stamp.english_company_text = "SHANGHAI TEST TECHNOLOGY CO.,LTD."
        stamp.center_lines = ("合同专用章", "业务办理")
        stamp.blank_background_probability = 1.0
        stamp.document_interference_probability = 0.0
        stamp.foreground_occlusion_probability = 0.0
        stamp.partial_crop_probability = 0.0
        stamp.large_rotation_probability = large_rotation_probability
        stamp.ink_degradation_probability = 0.0
        stamp.draw_stamp()
        stamp.join_stamp()
        return stamp

    def test_all_layouts_emit_aligned_nonempty_maps(self) -> None:
        cases = (
            ("standard", "circle"),
            ("bilingual_ring", "ellipse"),
            ("oval_service", "ellipse"),
            ("standard", "rectangle"),
            ("multiline_center", "circle"),
        )
        for seed, (layout, shape) in enumerate(cases, 100):
            with self.subTest(layout=layout, shape=shape):
                stamp = self._build_stamp(layout, shape, seed)
                annotation = stamp.build_spatial_annotation()
                detail_annotation = stamp.build_spatial_detail_annotation()
                text_mask, stamp_mask, text_heatmap, stamp_heatmap = (
                    annotation.split()
                )
                (
                    character_centers,
                    reading_progress,
                    first_character,
                    last_character,
                ) = detail_annotation.split()

                self.assertEqual(annotation.mode, "RGBA")
                self.assertEqual(annotation.size, stamp.joined_image.size)
                self.assertEqual(detail_annotation.mode, "RGBA")
                self.assertEqual(detail_annotation.size, stamp.joined_image.size)
                for channel in annotation.split():
                    self.assertIsNotNone(channel.getbbox())
                for channel in detail_annotation.split():
                    self.assertIsNotNone(channel.getbbox())
                self.assertLessEqual(set(text_mask.getdata()), {0, 255})
                self.assertLessEqual(set(stamp_mask.getdata()), {0, 255})
                self.assertTrue(
                    all(
                        not text_value or stamp_value
                        for text_value, stamp_value in zip(
                            text_mask.getdata(), stamp_mask.getdata()
                        )
                    )
                )
                self.assertGreater(len(set(text_heatmap.getdata())), 2)
                self.assertGreater(len(set(stamp_heatmap.getdata())), 2)
                self.assertGreater(len(set(character_centers.getdata())), 2)
                self.assertGreater(len(set(reading_progress.getdata())), 3)
                self.assertGreater(len(set(first_character.getdata())), 2)
                self.assertGreater(len(set(last_character.getdata())), 2)
                self.assertNotEqual(
                    first_character.tobytes(),
                    last_character.tobytes(),
                )

    def test_annotation_is_saved_as_separate_lossless_png(self) -> None:
        stamp = self._build_stamp("standard", "circle", 2048)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            stamp.save_name = "42"
            stamp.save_path = root / "images" / "00000"
            stamp.spatial_annotation_path = root / "spatial" / "00000"
            stamp.save_join_stamp()
            stamp.save_spatial_annotation()

            image_path = stamp.save_path / "42.jpg"
            annotation_path = stamp.spatial_annotation_path / "42.png"
            detail_path = stamp.spatial_annotation_path / "42.detail.png"
            self.assertTrue(image_path.is_file())
            self.assertTrue(annotation_path.is_file())
            self.assertTrue(detail_path.is_file())
            with Image.open(image_path) as image, Image.open(
                annotation_path
            ) as annotation, Image.open(detail_path) as detail_annotation:
                self.assertEqual(annotation.mode, "RGBA")
                self.assertEqual(annotation.size, image.size)
                self.assertEqual(detail_annotation.mode, "RGBA")
                self.assertEqual(detail_annotation.size, image.size)

    def test_large_rotation_keeps_spatial_annotations_aligned(self) -> None:
        stamp = self._build_stamp(
            "standard",
            "circle",
            4096,
            large_rotation_probability=1.0,
        )
        self.assertGreaterEqual(
            abs(stamp.actual_rotation_degrees),
            LARGE_ROTATION_MIN_DEGREES,
        )
        self.assertLessEqual(
            abs(stamp.actual_rotation_degrees),
            LARGE_ROTATION_MAX_DEGREES,
        )
        self.assertEqual(
            stamp.build_spatial_annotation().size,
            stamp.joined_image.size,
        )
        self.assertEqual(
            stamp.build_spatial_detail_annotation().size,
            stamp.joined_image.size,
        )

    def test_spatial_annotations_are_enabled_by_default(self) -> None:
        args = build_parser().parse_args([])
        self.assertTrue(args.spatial_annotations)
        self.assertIsNone(args.spatial_annotation_output)


if __name__ == "__main__":
    unittest.main()

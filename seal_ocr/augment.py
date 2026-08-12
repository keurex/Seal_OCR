"""预训练与真实数据后训练的图像增强。"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Callable

# imgaug 0.26~ 内部仍使用已弃用的 PiecewiseAffineTransform.estimate()，
# 大量 FutureWarning 会刷屏。我们不能在运行时改第三方库源码，统一在这里屏蔽。
warnings.filterwarnings(
    "ignore",
    message=r".*`estimate` is deprecated.*PiecewiseAffineTransform\.from_estimate.*",
    category=FutureWarning,
)

import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image

from seal_ocr.color import apply_color_robustness
from seal_ocr.degradation import DegradationConfig, apply_degradation
from seal_ocr.spatial_annotations import (
    SpatialAnnotationBundle,
    ensure_spatial_annotation_bundle,
)

try:
    import cv2

    # torchrun 下每个 DataLoader worker 不再额外创建 OpenCV 线程池，避免 128 核
    # 机器出现数百个竞争线程。
    cv2.setNumThreads(0)
except ImportError:
    pass


def _sometimes(probability, augmenter):
    return iaa.Sometimes(probability, augmenter)


# 合成预训练：生成器已经包含困难背景、缺墨和部分章。在线阶段只补充中等强度
# 的扫描/复印退化，避免双环小字尚未学清就被多重模糊、压缩、缺失同时破坏。
PRETRAIN_AUGMENTER = iaa.Sequential(
    [
        _sometimes(
            0.68,
            iaa.Affine(
                scale={"x": (0.90, 1.04), "y": (0.90, 1.04)},
                translate_percent={"x": (-0.035, 0.035), "y": (-0.035, 0.035)},
                rotate=(-6, 6),
                shear=(-2, 2),
                mode="edge",
            ),
        ),
        _sometimes(0.02, iaa.Affine(rotate=(-180, 180), mode="edge")),
        _sometimes(0.08, iaa.PerspectiveTransform(scale=(0.004, 0.024))),
        _sometimes(0.06, iaa.PiecewiseAffine(scale=(0.003, 0.012))),
        iaa.SomeOf(
            (0, 3),
            [
                _sometimes(0.55, iaa.GaussianBlur(sigma=(0.1, 1.0))),
                _sometimes(0.14, iaa.MotionBlur(k=(3, 5), angle=(-45, 45))),
                _sometimes(0.16, iaa.AverageBlur(k=(2, 3))),
                _sometimes(
                    0.55,
                    iaa.AdditiveGaussianNoise(scale=(0, 0.020 * 255)),
                ),
                _sometimes(0.50, iaa.JpegCompression(compression=(5, 45))),
                _sometimes(0.65, iaa.LinearContrast((0.72, 1.24))),
                _sometimes(0.65, iaa.Multiply((0.70, 1.16))),
                _sometimes(
                    0.25,
                    iaa.CoarseDropout(
                        p=(0.001, 0.007),
                        size_percent=(0.003, 0.014),
                        per_channel=False,
                    ),
                ),
                _sometimes(0.12, iaa.SaltAndPepper(p=(0.001, 0.004))),
                _sometimes(
                    0.35,
                    iaa.AddToHueAndSaturation((-7, 7), per_channel=False),
                ),
            ],
            random_order=True,
        ),
    ],
    random_order=False,
)


# 17 字以上公司名的单字弧长更短、笔画更细。仍保留轻度扫描变化，但不再把
# 多种破坏叠加到同一张图；否则标签要求输出完整名称，而视觉证据已经不可恢复。
LONG_PRETRAIN_AUGMENTER = iaa.Sequential(
    [
        _sometimes(
            0.55,
            iaa.Affine(
                scale={"x": (0.94, 1.03), "y": (0.94, 1.03)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-4, 4),
                shear=(-1, 1),
                mode="edge",
            ),
        ),
        _sometimes(0.04, iaa.PerspectiveTransform(scale=(0.003, 0.015))),
        iaa.SomeOf(
            (0, 2),
            [
                _sometimes(0.48, iaa.GaussianBlur(sigma=(0.1, 0.65))),
                _sometimes(
                    0.42,
                    iaa.AdditiveGaussianNoise(scale=(0, 0.012 * 255)),
                ),
                _sometimes(0.42, iaa.JpegCompression(compression=(5, 30))),
                _sometimes(0.55, iaa.LinearContrast((0.82, 1.16))),
                _sometimes(0.55, iaa.Multiply((0.80, 1.12))),
                _sometimes(
                    0.10,
                    iaa.CoarseDropout(
                        p=(0.001, 0.004),
                        size_percent=(0.003, 0.010),
                        per_channel=False,
                    ),
                ),
            ],
            random_order=True,
        ),
    ],
    random_order=False,
)


# 真实业务后训练：保留真实扫描件本身的退化，只施加温和扰动，防止几百张
# 新增标注被增强噪声淹没。
FINETUNE_AUGMENTER = iaa.Sequential(
    [
        _sometimes(
            0.65,
            iaa.Affine(
                scale={"x": (0.94, 1.03), "y": (0.94, 1.03)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-3, 3),
                mode="edge",
            ),
        ),
        _sometimes(0.01, iaa.Affine(rotate=(-180, 180), mode="edge")),
        iaa.SomeOf(
            (0, 2),
            [
                _sometimes(0.45, iaa.GaussianBlur(sigma=(0.1, 0.8))),
                _sometimes(
                    0.35,
                    iaa.AdditiveGaussianNoise(scale=(0, 0.015 * 255)),
                ),
                _sometimes(0.35, iaa.JpegCompression(compression=(5, 30))),
                _sometimes(0.45, iaa.LinearContrast((0.85, 1.15))),
                _sometimes(0.45, iaa.Multiply((0.82, 1.12))),
                _sometimes(
                    0.18,
                    iaa.CoarseDropout(
                        p=(0.001, 0.006),
                        size_percent=(0.003, 0.012),
                        per_channel=False,
                    ),
                ),
            ],
            random_order=True,
        ),
    ],
    random_order=False,
)


def _apply(
    augmenter,
    image_data: Image.Image,
    spatial_annotation: Image.Image | SpatialAnnotationBundle | None = None,
):
    array = np.asarray(image_data.convert("RGB"))
    if spatial_annotation is None:
        augmented = augmenter(image=array)
        return Image.fromarray(augmented)

    annotation_bundle = ensure_spatial_annotation_bundle(spatial_annotation)
    primary_array = np.asarray(annotation_bundle.primary)
    if primary_array.shape[:2] != array.shape[:2]:
        raise ValueError(
            "空间标注与图片尺寸不一致: "
            f"image={array.shape[:2]}, annotation={primary_array.shape[:2]}"
        )
    detail_array = None
    if annotation_bundle.detail is not None:
        detail_array = np.asarray(annotation_bundle.detail)
        if detail_array.shape[:2] != array.shape[:2]:
            raise ValueError(
                "空间细节标注与图片尺寸不一致: "
                f"image={array.shape[:2]}, detail={detail_array.shape[:2]}"
            )
    segmentation_maps = SegmentationMapsOnImage(
        (primary_array[:, :, :2] >= 128).astype(np.int32),
        shape=array.shape,
    )
    heatmap_channels = [primary_array[:, :, 2:]]
    if detail_array is not None:
        heatmap_channels.append(detail_array)
    heatmaps = HeatmapsOnImage(
        np.concatenate(heatmap_channels, axis=-1).astype(np.float32) / 255.0,
        shape=array.shape,
        min_value=0.0,
        max_value=1.0,
    )
    augmented_image, augmented_masks, augmented_heatmaps = augmenter(
        image=array,
        segmentation_maps=segmentation_maps,
        heatmaps=heatmaps,
    )
    mask_array = np.asarray(augmented_masks.get_arr())
    heatmap_array = np.asarray(augmented_heatmaps.get_arr())
    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, None]
    if heatmap_array.ndim == 2:
        heatmap_array = heatmap_array[:, :, None]
    expected_heatmap_channels = 6 if detail_array is not None else 2
    if (
        mask_array.shape[-1] != 2
        or heatmap_array.shape[-1] != expected_heatmap_channels
    ):
        raise RuntimeError(
            "空间增强后通道数量异常: "
            f"mask={mask_array.shape}, heatmap={heatmap_array.shape}"
        )
    augmented_primary = np.concatenate(
        [
            (mask_array > 0).astype(np.uint8) * 255,
            np.rint(
                np.clip(heatmap_array[:, :, :2], 0.0, 1.0) * 255
            ).astype(np.uint8),
        ],
        axis=-1,
    )
    augmented_detail = None
    if detail_array is not None:
        augmented_detail = np.rint(
            np.clip(heatmap_array[:, :, 2:], 0.0, 1.0) * 255
        ).astype(np.uint8)
    return (
        Image.fromarray(augmented_image),
        SpatialAnnotationBundle(
            primary=Image.fromarray(augmented_primary, mode="RGBA"),
            detail=(
                Image.fromarray(augmented_detail, mode="RGBA")
                if augmented_detail is not None
                else None
            ),
        ),
    )


def _apply_color_to_result(result, probability: float):
    if isinstance(result, tuple):
        image, annotation = result
        return apply_color_robustness(image, probability=probability), annotation
    return apply_color_robustness(result, probability=probability)


def aug_pretrain(
    image_data: Image.Image,
    label_length: int | None = None,
    spatial_annotation: Image.Image | SpatialAnnotationBundle | None = None,
    degradation_config: DegradationConfig | None = None,
):
    augmenter = (
        LONG_PRETRAIN_AUGMENTER
        if label_length is not None and label_length >= 17
        else PRETRAIN_AUGMENTER
    )
    result = _apply(augmenter, image_data, spatial_annotation)
    if isinstance(result, tuple):
        image, annotation = result
        image, _ = apply_degradation(
            image,
            annotation.primary,
            label_length=label_length,
            config=degradation_config or DegradationConfig(),
        )
        result = image, annotation
    return _apply_color_to_result(result, 0.40)


def aug_finetune(
    image_data: Image.Image,
    label_length: int | None = None,
    spatial_annotation: Image.Image | SpatialAnnotationBundle | None = None,
):
    # 真实样本较少，概率更温和，避免完全覆盖原始扫描分布。
    return _apply_color_to_result(
        _apply(FINETUNE_AUGMENTER, image_data, spatial_annotation),
        0.25,
    )


def get_augmentation(
    name: str,
    *,
    degradation_config: DegradationConfig | None = None,
) -> Callable[..., Image.Image]:
    augmentations = {
        "pretrain": aug_pretrain,
        "finetune": aug_finetune,
    }
    if name not in augmentations:
        raise ValueError(
            f"未知增强方案 {name!r}，可选值: {', '.join(sorted(augmentations))}"
        )
    if name == "pretrain":
        return partial(
            aug_pretrain,
            degradation_config=(
                degradation_config or DegradationConfig()
            ),
        )
    return augmentations[name]

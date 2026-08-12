"""训练、评测和部署共用的图片尺寸处理。"""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageDraw, ImageStat

from seal_ocr.spatial_annotations import (
    SpatialAnnotationBundle,
    ensure_spatial_annotation_bundle,
)


RESIZE_MODES = ("stretch", "letterbox")


def processor_image_size(processor) -> Tuple[int, int]:
    """以 ``(width, height)`` 返回 TrOCR processor 的目标尺寸。"""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = processor.feature_extractor
    size = image_processor.size
    if isinstance(size, int):
        return size, size
    if "width" in size and "height" in size:
        return int(size["width"]), int(size["height"])
    if "shortest_edge" in size:
        edge = int(size["shortest_edge"])
        return edge, edge
    raise ValueError(f"无法识别 processor 图片尺寸配置: {size!r}")


def _border_background_color(image: Image.Image) -> Tuple[int, int, int]:
    """用裁片四周像素的中位数估计纸张底色。"""
    image = image.convert("RGB")
    width, height = image.size
    border_width = max(1, min(width, height) // 40)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, 0, width - 1, border_width - 1), fill=255)
    draw.rectangle(
        (0, height - border_width, width - 1, height - 1),
        fill=255,
    )
    draw.rectangle((0, 0, border_width - 1, height - 1), fill=255)
    draw.rectangle(
        (width - border_width, 0, width - 1, height - 1),
        fill=255,
    )
    return tuple(
        int(value) for value in ImageStat.Stat(image, mask=mask).median[:3]
    )


def prepare_image_for_processor(
    image: Image.Image,
    processor,
    resize_mode: str,
) -> Image.Image:
    """在 processor 前处理图片，避免非方形印章被强行拉伸。

    ``stretch`` 保留历史行为；``letterbox`` 等比例缩放后使用裁片边缘估计的
    纸张底色补齐到模型尺寸。补齐后 processor 不再改变长宽比。
    """
    if resize_mode not in RESIZE_MODES:
        raise ValueError(
            f"未知 resize_mode={resize_mode!r}，可选值: {RESIZE_MODES}"
        )
    image = image.convert("RGB")
    if resize_mode == "stretch":
        return image

    target_width, target_height = processor_image_size(processor)
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"非法图片尺寸: {image.size}")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = Image.new(
        "RGB",
        (target_width, target_height),
        _border_background_color(image),
    )
    canvas.paste(
        resized,
        (
            (target_width - resized_width) // 2,
            (target_height - resized_height) // 2,
        ),
    )
    return canvas


def _prepare_spatial_rgba_for_processor(
    annotation: Image.Image,
    processor,
    resize_mode: str,
    *,
    binary_channel_count: int,
) -> Image.Image:
    """用 OCR 图片的几何规则处理一张 RGBA 标注。"""
    if resize_mode not in RESIZE_MODES:
        raise ValueError(
            f"未知 resize_mode={resize_mode!r}，可选值: {RESIZE_MODES}"
        )
    annotation = annotation.convert("RGBA")
    source_width, source_height = annotation.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"非法空间标注尺寸: {annotation.size}")

    target_width, target_height = processor_image_size(processor)
    if resize_mode == "stretch":
        resized_width, resized_height = target_width, target_height
        offset = (0, 0)
    else:
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        offset = (
            (target_width - resized_width) // 2,
            (target_height - resized_height) // 2,
        )

    channels = annotation.split()
    if not 0 <= binary_channel_count <= len(channels):
        raise ValueError(
            "binary_channel_count 超出 RGBA 通道范围: "
            f"{binary_channel_count}"
        )
    resized_channels = []
    for index, channel in enumerate(channels):
        resample = (
            Image.Resampling.NEAREST
            if index < binary_channel_count
            else Image.Resampling.BILINEAR
        )
        resized = channel.resize(
            (resized_width, resized_height),
            resample=resample,
        )
        canvas = Image.new("L", (target_width, target_height), 0)
        canvas.paste(resized, offset)
        if index < binary_channel_count:
            canvas = canvas.point(lambda value: 255 if value >= 128 else 0)
        resized_channels.append(canvas)
    return Image.merge("RGBA", resized_channels)


def prepare_spatial_annotation_for_processor(
    annotation: Image.Image,
    processor,
    resize_mode: str,
) -> Image.Image:
    """处理主标注；R/G 为 mask，B/A 为连续 heatmap。"""
    return _prepare_spatial_rgba_for_processor(
        annotation,
        processor,
        resize_mode,
        binary_channel_count=2,
    )


def prepare_spatial_annotation_bundle_for_processor(
    annotation: Image.Image | SpatialAnnotationBundle,
    processor,
    resize_mode: str,
) -> SpatialAnnotationBundle:
    """同步处理两张空间标注。

    detail 的四个通道都是连续目标：字符中心、阅读进度和首尾位置，因此
    使用双线性缩放；主标注仍保持 R/G 最近邻、B/A 双线性。
    """
    bundle = ensure_spatial_annotation_bundle(annotation)
    primary = _prepare_spatial_rgba_for_processor(
        bundle.primary,
        processor,
        resize_mode,
        binary_channel_count=2,
    )
    detail = (
        _prepare_spatial_rgba_for_processor(
            bundle.detail,
            processor,
            resize_mode,
            binary_channel_count=0,
        )
        if bundle.detail is not None
        else None
    )
    return SpatialAnnotationBundle(primary=primary, detail=detail)


def resolve_resize_mode(requested_mode: str, model) -> str:
    """解析 CLI 的 auto，并以模型保存的训练模式作为部署权威值。"""
    if requested_mode != "auto":
        if requested_mode not in RESIZE_MODES:
            raise ValueError(f"未知 resize_mode={requested_mode!r}")
        return requested_mode
    configured_mode = getattr(model.config, "seal_resize_mode", None)
    if configured_mode in RESIZE_MODES:
        return configured_mode
    return "stretch"

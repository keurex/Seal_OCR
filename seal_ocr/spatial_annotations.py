"""八通道空间标注文件格式与加载。

每条标注由同目录、同基名的两张 RGBA PNG 组成：

主标注 ``name.png``：
    R 公司文字二值 mask；G 完整章形二值 mask；
    B 公司文字 Gaussian heatmap；A 完整章形 Gaussian heatmap。

细节标注 ``name.detail.png``：
    R 全部公司字符中心 heatmap；G 字符阅读进度图；
    B 首字符中心 heatmap；A 末字符中心 heatmap。

拆成两张无损 RGBA PNG，既控制体积，也让几何增强同步变换全部通道。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image


PRIMARY_SPATIAL_CHANNELS = (
    "company_text_binary_mask",
    "full_stamp_binary_mask",
    "company_text_gaussian_heatmap",
    "full_stamp_gaussian_heatmap",
)

DETAIL_SPATIAL_CHANNELS = (
    "company_character_center_gaussian_heatmap",
    "company_character_reading_progress",
    "company_first_character_gaussian_heatmap",
    "company_last_character_gaussian_heatmap",
)

SPATIAL_CHANNELS = PRIMARY_SPATIAL_CHANNELS + DETAIL_SPATIAL_CHANNELS


def spatial_detail_annotation_path(primary_path: Union[str, Path]) -> Path:
    """返回细节标注路径，例如 ``42.png -> 42.detail.png``。"""
    path = Path(primary_path)
    return path.with_name(f"{path.stem}.detail{path.suffix}")


@dataclass(frozen=True)
class SpatialAnnotationBundle:
    """一条样本的主空间标注和可选细节标注。"""

    primary: Image.Image
    detail: Image.Image | None = None

    def __post_init__(self) -> None:
        if self.primary.mode != "RGBA":
            raise ValueError(
                f"空间主标注必须为 RGBA，实际为 {self.primary.mode}"
            )
        if self.detail is not None:
            if self.detail.mode != "RGBA":
                raise ValueError(
                    f"空间细节标注必须为 RGBA，实际为 {self.detail.mode}"
                )
            if self.detail.size != self.primary.size:
                raise ValueError(
                    "空间主标注与细节标注尺寸不一致: "
                    f"primary={self.primary.size}, detail={self.detail.size}"
                )

    @property
    def size(self):
        return self.primary.size

    @property
    def channel_count(self) -> int:
        return len(SPATIAL_CHANNELS) if self.detail is not None else len(
            PRIMARY_SPATIAL_CHANNELS
        )


def ensure_spatial_annotation_bundle(
    annotation: Image.Image | SpatialAnnotationBundle,
) -> SpatialAnnotationBundle:
    """把单张主标注规范为 bundle。"""
    if isinstance(annotation, SpatialAnnotationBundle):
        return annotation
    if not isinstance(annotation, Image.Image):
        raise TypeError(
            "空间标注必须是 PIL.Image 或 SpatialAnnotationBundle，"
            f"实际为 {type(annotation).__name__}"
        )
    return SpatialAnnotationBundle(primary=annotation.convert("RGBA"))


def load_spatial_annotation_bundle(
    primary_path: Union[str, Path],
    *,
    require_detail: bool = False,
    load_detail: bool = True,
) -> SpatialAnnotationBundle:
    """加载一组空间标注；训练所需 detail 缺失时立即报错。"""
    primary_path = Path(primary_path)
    with Image.open(primary_path) as opened_primary:
        if opened_primary.mode != "RGBA":
            raise ValueError(
                f"空间主标注必须为 RGBA: {primary_path}, "
                f"实际为 {opened_primary.mode}"
            )
        primary = opened_primary.copy()

    if require_detail and not load_detail:
        raise ValueError("require_detail=True 时不能关闭 load_detail")
    detail_path = spatial_detail_annotation_path(primary_path)
    detail = None
    if load_detail and detail_path.is_file():
        with Image.open(detail_path) as opened_detail:
            if opened_detail.mode != "RGBA":
                raise ValueError(
                    f"空间细节标注必须为 RGBA: {detail_path}, "
                    f"实际为 {opened_detail.mode}"
                )
            detail = opened_detail.copy()
    elif require_detail:
        raise FileNotFoundError(
            "空间监督需要字符中心/阅读顺序细节标注，"
            f"但未找到 {detail_path}。请用当前生成器重新生成空间标注。"
        )

    bundle = SpatialAnnotationBundle(primary=primary, detail=detail)
    return bundle

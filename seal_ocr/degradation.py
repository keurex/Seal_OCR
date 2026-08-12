"""利用空间标注生成预训练用的笔画与文档退化。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_DIR = Path(__file__).resolve().parents[1] / "synthesis" / "fonts"
_FONT_CANDIDATES = tuple(
    sorted(
        path
        for path in _FONT_DIR.rglob("*")
        if path.suffix.lower() in {".ttf", ".ttc", ".otf"}
    )
) + (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)

_DOCUMENT_TEXT_SNIPPETS = (
    "购货单位名称",
    "销售方开户银行及账号",
    "项目名称及合同编号",
    "经办人签字确认",
    "审核日期",
    "金额合计人民币",
    "本页内容与原件一致",
    "付款条件及结算方式",
    "统一社会信用代码",
    "地址电话",
)

_ASCII_TEXT_SNIPPETS = (
    "INVOICE NO.",
    "CONTRACT DATE",
    "TOTAL AMOUNT",
    "ACCOUNT NUMBER",
    "APPROVED BY",
)


@dataclass(frozen=True)
class DegradationConfig:
    """预训练定向退化参数。

    消解只在公司文字笔画像素内降低印泥对比度，不生成连续白块。
    ``min_text_residual_ratio`` 是硬下限：即使最严重的像素也至少保留该比例的
    原始印泥对比度。前景手写内容只能是细线或小斑点，且与文字 mask 的交叠受限。
    """

    enabled: bool = True
    probability: float = 0.60
    text_dissolution_probability: float = 0.75
    background_clutter_probability: float = 0.70
    foreground_stroke_probability: float = 0.30
    max_text_dissolution_ratio: float = 0.70
    min_text_residual_ratio: float = 0.25
    max_foreground_text_overlap_ratio: float = 0.06

    def __post_init__(self) -> None:
        probability_fields = {
            "probability": self.probability,
            "text_dissolution_probability": (
                self.text_dissolution_probability
            ),
            "background_clutter_probability": (
                self.background_clutter_probability
            ),
            "foreground_stroke_probability": (
                self.foreground_stroke_probability
            ),
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 范围内")
        if not 0.0 <= self.max_text_dissolution_ratio <= 0.85:
            raise ValueError(
                "max_text_dissolution_ratio 必须在 [0, 0.85] 范围内"
            )
        if not 0.20 <= self.min_text_residual_ratio <= 1.0:
            raise ValueError(
                "min_text_residual_ratio 必须在 [0.20, 1.0] 范围内"
            )
        if not 0.0 <= self.max_foreground_text_overlap_ratio <= 0.12:
            raise ValueError(
                "max_foreground_text_overlap_ratio 必须在 [0, 0.12] 范围内"
            )


@dataclass(frozen=True)
class DegradationStats:
    """单张图片实际施加的退化摘要，供回归测试和预览使用。"""

    applied: bool = False
    text_dissolution_applied: bool = False
    background_clutter_applied: bool = False
    foreground_stroke_applied: bool = False
    text_dissolution_ratio: float = 0.0
    minimum_text_residual_ratio: float = 1.0
    clutter_coverage: float = 0.0
    foreground_text_overlap_ratio: float = 0.0


def _new_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is not None:
        return rng
    # DataLoader 会按 worker 设置 numpy 全局种子。从该状态再派生 Generator，
    # 避免 fork worker 获得同一序列，同时保持同种子训练可复现。
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
    return np.random.default_rng(seed)


def _read_annotation_masks(
    spatial_annotation: Image.Image,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if spatial_annotation.mode != "RGBA":
        raise ValueError(
            "定向退化要求 RGBA 空间标注，"
            f"实际为 {spatial_annotation.mode}"
        )
    if spatial_annotation.size != image_size:
        raise ValueError(
            "定向退化的图片与空间标注尺寸不一致: "
            f"image={image_size}, annotation={spatial_annotation.size}"
        )
    annotation = np.asarray(spatial_annotation)
    return annotation[:, :, 0] >= 128, annotation[:, :, 1] >= 128


def _blend(
    source: np.ndarray,
    replacement: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    return np.clip(
        source.astype(np.float32) * (1.0 - alpha)
        + replacement.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _stamp_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(ys):
        return 0, 0, width, height
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _estimate_paper_color(
    image_array: np.ndarray,
    stamp_mask: np.ndarray,
) -> np.ndarray:
    height, width = stamp_mask.shape
    x0, y0, x1, y1 = _stamp_bounds(stamp_mask)
    padding = max(6, round(min(height, width) * 0.035))
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    context = np.zeros_like(stamp_mask, dtype=bool)
    context[y0:y1, x0:x1] = True
    context &= ~stamp_mask
    pixels = image_array[context]
    if len(pixels) < 32:
        pixels = image_array.reshape(-1, 3)
    # 黑色单据字和少量彩色印泥不会明显拉偏各通道中位数。
    return np.median(pixels, axis=0)


def _coarse_random_field(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grid_height = max(3, round(height / 48))
    grid_width = max(3, round(width / 48))
    small = np.rint(
        rng.random((grid_height, grid_width)) * 255
    ).astype(np.uint8)
    field = Image.fromarray(small, mode="L").resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(field, dtype=np.float32) / 255.0


def sample_text_dissolution_alpha(
    text_mask: np.ndarray,
    *,
    max_dissolution_ratio: float,
    min_residual_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    """生成笔画内部的颗粒消解透明度。

    选择由像素级随机场主导，低频场只控制某些弧段更淡，因此不会形成覆盖整个
    单字的规则块。每个被选像素的透明度都不超过 ``1-min_residual_ratio``。
    """

    if text_mask.ndim != 2:
        raise ValueError("text_mask 必须为二维数组")
    text_mask = text_mask.astype(bool, copy=False)
    text_pixel_count = int(text_mask.sum())
    if text_pixel_count == 0 or max_dissolution_ratio <= 0:
        return np.zeros_like(text_mask, dtype=np.float32), 0.0, 1.0
    maximum_pixels = int(np.floor(text_pixel_count * max_dissolution_ratio))
    if maximum_pixels <= 0:
        return np.zeros_like(text_mask, dtype=np.float32), 0.0, 1.0

    height, width = text_mask.shape
    minimum_ratio = min(0.28, max_dissolution_ratio)
    target_ratio = float(
        rng.uniform(minimum_ratio, max_dissolution_ratio)
        if max_dissolution_ratio > minimum_ratio
        else max_dissolution_ratio
    )
    fine = rng.random((height, width), dtype=np.float32)
    coarse = _coarse_random_field(height, width, rng)
    score = fine * 0.72 + coarse * 0.28
    text_scores = score[text_mask]
    threshold = float(np.quantile(text_scores, 1.0 - target_ratio))
    selected = text_mask & (score >= threshold)

    # 严格裁掉浮点分位数并列导致的少量超额像素。
    selected_count = int(selected.sum())
    if selected_count > maximum_pixels:
        selected_points = np.argwhere(selected)
        selected_scores = score[selected]
        keep_indices = np.argpartition(
            selected_scores,
            -maximum_pixels,
        )[-maximum_pixels:]
        selected = np.zeros_like(text_mask, dtype=bool)
        kept_points = selected_points[keep_indices]
        selected[kept_points[:, 0], kept_points[:, 1]] = True
        selected_count = maximum_pixels

    maximum_alpha = 1.0 - min_residual_ratio
    normalized_score = np.clip((score - threshold) / 0.35, 0.0, 1.0)
    severity = 0.32 + normalized_score * max(0.0, maximum_alpha - 0.32)
    alpha = np.where(selected, np.minimum(severity, maximum_alpha), 0.0)
    alpha = alpha.astype(np.float32)
    actual_ratio = selected_count / text_pixel_count
    actual_minimum_residual = (
        float(1.0 - alpha[selected].max()) if selected_count else 1.0
    )
    return alpha, actual_ratio, actual_minimum_residual


def _apply_text_dissolution(
    image_array: np.ndarray,
    text_mask: np.ndarray,
    stamp_mask: np.ndarray,
    *,
    max_dissolution_ratio: float,
    min_residual_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    alpha, dissolution_ratio, actual_residual = (
        sample_text_dissolution_alpha(
            text_mask,
            max_dissolution_ratio=max_dissolution_ratio,
            min_residual_ratio=min_residual_ratio,
            rng=rng,
        )
    )
    if not np.any(alpha):
        return image_array, 0.0, 1.0
    paper_color = _estimate_paper_color(image_array, stamp_mask)
    replacement = np.empty_like(image_array)
    replacement[:, :] = np.rint(paper_color).astype(np.uint8)
    return (
        _blend(image_array, replacement, alpha),
        dissolution_ratio,
        actual_residual,
    )


@lru_cache(maxsize=64)
def _load_document_font(font_size: int):
    for font_path in _FONT_CANDIDATES:
        if not font_path.is_file():
            continue
        try:
            return ImageFont.truetype(str(font_path), font_size), True
        except OSError:
            continue
    return ImageFont.load_default(), False


def _random_document_text(
    rng: np.random.Generator,
    chinese_font_available: bool,
) -> str:
    prefix = str(
        rng.choice(
            _DOCUMENT_TEXT_SNIPPETS
            if chinese_font_available
            else _ASCII_TEXT_SNIPPETS
        )
    )
    suffix_type = float(rng.random())
    if suffix_type < 0.34:
        separator = "：" if chinese_font_available else ":"
        suffix = f"{separator}{int(rng.integers(1, 999999)):06d}"
    elif suffix_type < 0.68:
        suffix = (
            f" 20{int(rng.integers(18, 30)):02d}-"
            f"{int(rng.integers(1, 13)):02d}-"
            f"{int(rng.integers(1, 29)):02d}"
        )
    else:
        currency = "¥" if chinese_font_available else "$"
        suffix = f"  {currency}{float(rng.uniform(10, 999999)):,.2f}"
    return prefix + suffix


def _apply_document_clutter(
    image_array: np.ndarray,
    text_mask: np.ndarray,
    stamp_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    height, width = image_array.shape[:2]
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    if rng.random() < 0.68:
        line_alpha = int(rng.integers(65, 150))
        line_color = (
            int(rng.integers(0, 38)),
            int(rng.integers(0, 38)),
            int(rng.integers(0, 38)),
            line_alpha,
        )
        for _ in range(int(rng.integers(2, 8))):
            y = int(rng.integers(0, max(height, 1)))
            draw.line((0, y, width, y), fill=line_color, width=1)
        for _ in range(int(rng.integers(1, 5))):
            x = int(rng.integers(0, max(width, 1)))
            draw.line((x, 0, x, height), fill=line_color, width=1)

    text_line_count = int(rng.integers(5, 12))
    base_scale = min(height, width) / 384.0
    for _ in range(text_line_count):
        font_size = max(10, round(rng.uniform(13, 24) * base_scale))
        font, chinese_font_available = _load_document_font(font_size)
        draw.text(
            (
                int(rng.integers(-max(1, width // 6), max(2, width * 3 // 5))),
                int(rng.integers(-font_size, max(1, height))),
            ),
            _random_document_text(rng, chinese_font_available),
            font=font,
            fill=(
                int(rng.integers(0, 46)),
                int(rng.integers(0, 46)),
                int(rng.integers(0, 46)),
                int(rng.integers(95, 205)),
            ),
            stroke_width=1 if rng.random() < 0.10 else 0,
        )

    overlay_array = np.asarray(overlay, dtype=np.uint8)
    alpha = overlay_array[:, :, 3].astype(np.float32) / 255.0
    # 背景字视作印章下层内容：在印泥像素上显著衰减，并限制其在公司文字处
    # 的不透明度。这样允许前景/背景竞争，但不会把一个字整体涂黑。
    stamp_attenuation = float(rng.uniform(0.26, 0.48))
    alpha = np.where(stamp_mask, alpha * stamp_attenuation, alpha)
    # 最严重消解像素仍有 25% 印泥残留；此处再乘最多 20% 的下层黑字后，
    # 原印泥贡献仍至少约 20%，不会被背景字二次吃光。
    alpha = np.where(text_mask, np.minimum(alpha, 0.20), alpha)
    clutter_coverage = float(np.mean(alpha >= 0.08))
    return _blend(image_array, overlay_array[:, :, :3], alpha), clutter_coverage


def _sample_polyline(
    bounds: Tuple[int, int, int, int],
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], ...]:
    x0, y0, x1, y1 = bounds
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    horizontal = rng.random() < 0.72
    point_count = int(rng.integers(4, 8))
    if horizontal:
        start_x = int(rng.integers(x0, max(x0 + 1, x0 + width // 3)))
        end_x = int(rng.integers(max(start_x + 1, x0 + width // 2), x1 + 1))
        baseline = int(rng.integers(y0, y1 + 1))
        jitter = max(2, round(height * 0.055))
        return tuple(
            (
                round(start_x + (end_x - start_x) * index / (point_count - 1)),
                int(np.clip(baseline + rng.integers(-jitter, jitter + 1), y0, y1)),
            )
            for index in range(point_count)
        )
    start_y = int(rng.integers(y0, max(y0 + 1, y0 + height // 3)))
    end_y = int(rng.integers(max(start_y + 1, y0 + height // 2), y1 + 1))
    baseline = int(rng.integers(x0, x1 + 1))
    jitter = max(2, round(width * 0.055))
    return tuple(
        (
            int(np.clip(baseline + rng.integers(-jitter, jitter + 1), x0, x1)),
            round(start_y + (end_y - start_y) * index / (point_count - 1)),
        )
        for index in range(point_count)
    )


def _apply_fine_foreground_strokes(
    image_array: np.ndarray,
    text_mask: np.ndarray,
    stamp_mask: np.ndarray,
    *,
    max_text_overlap_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, bool]:
    height, width = text_mask.shape
    text_pixel_count = int(text_mask.sum())
    maximum_overlap = int(
        np.floor(text_pixel_count * max_text_overlap_ratio)
    )
    if text_pixel_count == 0 or maximum_overlap <= 0:
        return image_array, 0.0, False

    x0, y0, x1, y1 = _stamp_bounds(stamp_mask)
    x1 = max(x0, x1 - 1)
    y1 = max(y0, y1 - 1)
    accepted_mask = np.zeros((height, width), dtype=np.uint8)
    accepted_overlap = np.zeros_like(text_mask, dtype=bool)
    minimum_side = min(height, width)
    line_width = max(1, round(minimum_side / 384.0))

    for _ in range(int(rng.integers(1, 4))):
        candidate_image = Image.new("L", (width, height), 0)
        candidate_draw = ImageDraw.Draw(candidate_image)
        candidate_draw.line(
            _sample_polyline((x0, y0, x1, y1), rng),
            fill=int(rng.integers(145, 225)),
            width=line_width,
            joint="curve",
        )
        candidate = np.asarray(candidate_image, dtype=np.uint8)
        candidate_overlap = (candidate > 0) & text_mask & ~accepted_overlap
        if int(accepted_overlap.sum() + candidate_overlap.sum()) > maximum_overlap:
            continue
        accepted_mask = np.maximum(accepted_mask, candidate)
        accepted_overlap |= candidate_overlap

    # 小斑点模拟签字墨点或扫描脏点；半径最多约 2 px（以 384 输入计）。
    dot_image = Image.new("L", (width, height), 0)
    dot_draw = ImageDraw.Draw(dot_image)
    for _ in range(int(rng.integers(3, 11))):
        center_x = int(rng.integers(x0, x1 + 1))
        center_y = int(rng.integers(y0, y1 + 1))
        radius = max(1, round(rng.uniform(0.7, 1.8) * minimum_side / 384.0))
        dot_draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=int(rng.integers(110, 205)),
        )
    dots = np.asarray(dot_image, dtype=np.uint8)
    dot_overlap = (dots > 0) & text_mask & ~accepted_overlap
    remaining_overlap = maximum_overlap - int(accepted_overlap.sum())
    if int(dot_overlap.sum()) <= remaining_overlap:
        accepted_mask = np.maximum(accepted_mask, dots)
        accepted_overlap |= dot_overlap

    if not np.any(accepted_mask):
        return image_array, 0.0, False
    alpha = accepted_mask.astype(np.float32) / 255.0
    dark_color = np.array(
        [int(rng.integers(0, 28)) for _ in range(3)],
        dtype=np.uint8,
    )
    replacement = np.empty_like(image_array)
    replacement[:, :] = dark_color
    overlap_ratio = float(accepted_overlap.sum()) / text_pixel_count
    return _blend(image_array, replacement, alpha), overlap_ratio, True


def apply_degradation(
    image: Image.Image,
    spatial_annotation: Image.Image,
    *,
    label_length: Optional[int] = None,
    config: Optional[DegradationConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Image.Image, DegradationStats]:
    """只改变图像，不改变理想空间标注。"""

    config = config or DegradationConfig()
    if not config.enabled:
        return image, DegradationStats()
    generator = _new_rng(rng)
    text_mask, stamp_mask = _read_annotation_masks(
        spatial_annotation,
        image.size,
    )
    if generator.random() >= config.probability:
        return image, DegradationStats()

    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    text_dissolution_applied = (
        bool(text_mask.any())
        and generator.random() < config.text_dissolution_probability
    )
    dissolution_ratio = 0.0
    minimum_residual = 1.0
    if text_dissolution_applied:
        # 长名称的单字更细，降低消解覆盖率并把最低残留提高到 30%。
        coverage_scale = 1.0
        minimum_residual = config.min_text_residual_ratio
        if label_length is not None and label_length >= 20:
            coverage_scale = 0.70
            minimum_residual = max(minimum_residual, 0.30)
        elif label_length is not None and label_length >= 17:
            coverage_scale = 0.80
            minimum_residual = max(minimum_residual, 0.30)
        image_array, dissolution_ratio, minimum_residual = (
            _apply_text_dissolution(
                image_array,
                text_mask,
                stamp_mask,
                max_dissolution_ratio=(
                    config.max_text_dissolution_ratio * coverage_scale
                ),
                min_residual_ratio=minimum_residual,
                rng=generator,
            )
        )
        text_dissolution_applied = dissolution_ratio > 0

    clutter_applied = (
        generator.random() < config.background_clutter_probability
    )
    clutter_coverage = 0.0
    if clutter_applied:
        image_array, clutter_coverage = _apply_document_clutter(
            image_array,
            text_mask,
            stamp_mask,
            generator,
        )

    foreground_stroke_applied = (
        generator.random() < config.foreground_stroke_probability
    )
    foreground_overlap = 0.0
    if foreground_stroke_applied:
        image_array, foreground_overlap, foreground_stroke_applied = (
            _apply_fine_foreground_strokes(
                image_array,
                text_mask,
                stamp_mask,
                max_text_overlap_ratio=(
                    config.max_foreground_text_overlap_ratio
                ),
                rng=generator,
            )
        )

    applied = (
        text_dissolution_applied
        or clutter_applied
        or foreground_stroke_applied
    )
    return (
        Image.fromarray(image_array, mode="RGB"),
        DegradationStats(
            applied=applied,
            text_dissolution_applied=text_dissolution_applied,
            background_clutter_applied=clutter_applied,
            foreground_stroke_applied=foreground_stroke_applied,
            text_dissolution_ratio=dissolution_ratio,
            minimum_text_residual_ratio=minimum_residual,
            clutter_coverage=clutter_coverage,
            foreground_text_overlap_ratio=foreground_overlap,
        ),
    )

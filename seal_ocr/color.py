"""与印章颜色无关的图像变换。

颜色是印章 OCR 的干扰变量：同一套文字和章形可能是红、蓝、紫、黑，也可能在
扫描后只剩灰度。这里的变换只改变颜色表达，不改变几何位置和文字笔画。
"""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _grayscale_rgb(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image.convert("RGB"))
    return Image.merge("RGB", (gray, gray, gray))


def _shift_hue_and_saturation(image: Image.Image) -> Image.Image:
    hue, saturation, value = image.convert("RGB").convert("HSV").split()
    hue_shift = random.randint(24, 232)
    hue = hue.point(lambda pixel: (pixel + hue_shift) % 256)
    saturation = ImageEnhance.Brightness(saturation).enhance(
        random.uniform(0.30, 1.45)
    )
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


def _shuffle_channels(image: Image.Image) -> Image.Image:
    channels = list(image.convert("RGB").split())
    order = [0, 1, 2]
    while order == [0, 1, 2]:
        random.shuffle(order)
    return Image.merge("RGB", tuple(channels[index] for index in order))


def apply_color_robustness(
    image: Image.Image,
    probability: float,
) -> Image.Image:
    """随机去色或换色，阻断“颜色 -> 公司名”的捷径。

    变换保留原始亮度结构和空间结构，因此圆弧、文字笔画、边框和遮挡位置不变。
    """

    if not 0 <= probability <= 1:
        raise ValueError("颜色鲁棒增强概率必须在 0 到 1 之间")
    image = image.convert("RGB")
    if random.random() >= probability:
        return image

    mode = random.random()
    if mode < 0.50:
        # 不总是完全灰度，保留部分低饱和扫描件。
        return Image.blend(
            image,
            _grayscale_rgb(image),
            random.uniform(0.70, 1.0),
        )
    if mode < 0.85:
        return _shift_hue_and_saturation(image)
    return _shuffle_channels(image)


def make_color_invariant_view(image: Image.Image) -> Image.Image:
    """生成与原图几何完全一致的灰度笔画视图，用于一致性训练。"""

    gray = ImageOps.grayscale(image.convert("RGB"))
    if random.random() < 0.50:
        gray = ImageOps.autocontrast(gray, cutoff=random.randint(0, 2))
    gray = ImageEnhance.Contrast(gray).enhance(random.uniform(0.82, 1.24))
    if random.random() < 0.25:
        gray = gray.filter(
            ImageFilter.UnsharpMask(
                radius=random.uniform(0.45, 0.90),
                percent=random.randint(35, 80),
                threshold=random.randint(2, 5),
            )
        )
    return Image.merge("RGB", (gray, gray, gray))

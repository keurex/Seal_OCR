"""TrOCR 图片与字符标签数据集。"""

from __future__ import annotations

import json
import os
from hashlib import blake2b
from math import ceil
from collections import defaultdict
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from seal_ocr.color import make_color_invariant_view
from seal_ocr.data import (
    TrainingSample,
    normalize_company_name,
)
from seal_ocr.image import (
    prepare_image_for_processor,
    prepare_spatial_annotation_bundle_for_processor,
)
from seal_ocr.spatial_annotations import (
    SPATIAL_CHANNELS,
    SpatialAnnotationBundle,
    ensure_spatial_annotation_bundle,
    load_spatial_annotation_bundle,
)


def build_spatial_targets(
    prepared_annotation: Image.Image | SpatialAnnotationBundle,
    target_size: Tuple[int, int],
):
    """把处理后的全分辨率标注下采样到 encoder patch 网格。

    二值 mask 用 max pooling 保住细笔画；普通 heatmap 用 area pooling。
    阅读进度只在公司文字前景内做加权均值，避免细笔画被整块 patch 的背景
    稀释到接近 0。
    """
    grid_height, grid_width = (int(value) for value in target_size)
    bundle = ensure_spatial_annotation_bundle(prepared_annotation)
    primary_array = np.array(bundle.primary, dtype=np.float32, copy=True)
    primary = torch.from_numpy(primary_array).permute(2, 0, 1) / 255.0
    text_mask_full = (primary[0:1] >= 0.5).float()
    mask_targets = F.adaptive_max_pool2d(
        (primary[:2] >= 0.5).float().unsqueeze(0),
        (grid_height, grid_width),
    ).squeeze(0)
    primary_heatmaps = F.interpolate(
        primary[2:].unsqueeze(0),
        size=(grid_height, grid_width),
        mode="area",
    ).squeeze(0)
    targets = [mask_targets, primary_heatmaps]

    if bundle.detail is None:
        raise ValueError("八通道空间监督缺少 detail 标注；请重新生成数据")
    detail_array = np.array(bundle.detail, dtype=np.float32, copy=True)
    detail = torch.from_numpy(detail_array).permute(2, 0, 1) / 255.0
    detail_heatmaps = F.interpolate(
        detail.unsqueeze(0),
        size=(grid_height, grid_width),
        mode="area",
    ).squeeze(0)

    progress_numerator = F.interpolate(
        (detail[1:2] * text_mask_full).unsqueeze(0),
        size=(grid_height, grid_width),
        mode="area",
    ).squeeze(0)
    progress_denominator = F.interpolate(
        text_mask_full.unsqueeze(0),
        size=(grid_height, grid_width),
        mode="area",
    ).squeeze(0)
    reading_progress = torch.where(
        progress_denominator > 1e-6,
        progress_numerator / progress_denominator.clamp_min(1e-6),
        torch.zeros_like(progress_numerator),
    )
    detail_heatmaps[1:2] = reading_progress
    targets.append(detail_heatmaps)

    return torch.cat(targets, dim=0).clamp(0.0, 1.0)


def identity_image_transform(
    image,
    label_length=None,
    spatial_annotation=None,
    **kwargs,
):
    """不做图像增强，同时保持空间标注与图像同步传递。

    评估集通常不需要增强，但带合成空间标注的 eval 样本仍会把
    ``spatial_annotation`` 传入 transformer。旧的二参数 lambda 在这里会让
    DataLoader worker 直接抛 ``unexpected keyword argument``，导致训练在第一
    个 eval step 前中止。
    """
    del label_length, kwargs
    if spatial_annotation is None:
        return image
    return image, spatial_annotation


class trocrDataset(Dataset):
    """
    TrOCR 训练数据集。

    推荐传入 ``TrainingSample``，这样训练前已经完成标签规范化与图片审计。
    为兼容旧调用，也仍支持传入图片路径字符串。
    """

    def __init__(
        self,
        paths: Sequence[Union[str, TrainingSample]],
        processor,
        max_target_length: int = 128,
        transformer=identity_image_transform,
        color_invariant_view: bool = False,
        resize_mode: str = "stretch",
        spatial_target_size: Optional[Tuple[int, int]] = None,
        load_spatial_annotation_for_augmentation: bool = False,
    ):
        self.paths = list(paths)
        self.processor = processor
        self.transformer = transformer
        self.color_invariant_view = color_invariant_view
        self.resize_mode = resize_mode
        self.spatial_target_size = spatial_target_size
        self.spatial_target_channels = len(SPATIAL_CHANNELS)
        self.load_spatial_annotation_for_augmentation = bool(
            load_spatial_annotation_for_augmentation
        )
        self.max_target_length = max_target_length
        self.vocab = processor.tokenizer.get_vocab()

        if not self.paths:
            raise ValueError("数据集不能为空")
        if max_target_length < 3:
            raise ValueError("max_target_length 至少为 3")
        if spatial_target_size is not None and (
            len(spatial_target_size) != 2
            or any(int(value) <= 0 for value in spatial_target_size)
        ):
            raise ValueError("spatial_target_size 必须为两个正整数")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_label(self, item: Union[str, TrainingSample]) -> str:
        if isinstance(item, TrainingSample):
            return item.label

        txt_file = os.path.splitext(item)[0] + ".txt"
        with open(txt_file, encoding="utf-8") as file:
            text = file.read()

        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return normalize_company_name(text)

    def __getitem__(self, idx: int):
        item = self.paths[idx]
        image_file = item.image_path if isinstance(item, TrainingSample) else item
        text = self._load_label(item)

        with Image.open(image_file) as opened_image:
            image = opened_image.convert("RGB")

        spatial_annotation = None
        spatial_annotation_path = (
            item.spatial_annotation_path
            if isinstance(item, TrainingSample)
            and (
                self.spatial_target_size is not None
                or self.load_spatial_annotation_for_augmentation
            )
            else None
        )
        if spatial_annotation_path is not None:
            spatial_annotation = load_spatial_annotation_bundle(
                spatial_annotation_path,
                require_detail=self.spatial_target_size is not None,
                load_detail=self.spatial_target_size is not None,
            )
            if spatial_annotation.size != image.size:
                raise ValueError(
                    "空间标注与图片尺寸不一致: "
                    f"image={image_file} {image.size}, "
                    f"annotation={spatial_annotation_path} "
                    f"{spatial_annotation.size}"
                )

        if spatial_annotation is not None:
            transformed = self.transformer(
                image,
                label_length=len(text),
                spatial_annotation=spatial_annotation,
            )
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise RuntimeError(
                    "带空间标注的数据增强必须返回 (image, annotation)"
                )
            image, spatial_annotation = transformed
        else:
            image = self.transformer(image, label_length=len(text))

        prepared_annotation = None
        if (
            spatial_annotation is not None
            and self.spatial_target_size is not None
        ):
            prepared_annotation = prepare_spatial_annotation_bundle_for_processor(
                spatial_annotation,
                processor=self.processor,
                resize_mode=self.resize_mode,
            )
        image = prepare_image_for_processor(
            image,
            processor=self.processor,
            resize_mode=self.resize_mode,
        )
        pixel_values = self.processor(
            image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)
        invariant_pixel_values = None
        if self.color_invariant_view:
            # 先完成同一套几何/扫描增强，再只改变颜色表达，保证两路视图的
            # 章形、文字位置和笔画严格对齐。
            invariant_image = make_color_invariant_view(image)
            invariant_pixel_values = self.processor(
                invariant_image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)

        labels = encode_text(
            text,
            max_target_length=self.max_target_length,
            vocab=self.vocab,
        )
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels = [label if label != pad_token_id else -100 for label in labels]

        result = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if invariant_pixel_values is not None:
            result["color_invariant_pixel_values"] = invariant_pixel_values
        if self.spatial_target_size is not None:
            grid_height, grid_width = self.spatial_target_size
            if prepared_annotation is None:
                spatial_targets = torch.zeros(
                    (
                        self.spatial_target_channels,
                        grid_height,
                        grid_width,
                    ),
                    dtype=torch.float32,
                )
                has_spatial_annotation = False
            else:
                spatial_targets = build_spatial_targets(
                    prepared_annotation,
                    (grid_height, grid_width),
                )
                has_spatial_annotation = True
            result["spatial_targets"] = spatial_targets
            result["has_spatial_annotation"] = torch.tensor(
                has_spatial_annotation,
                dtype=torch.bool,
            )
        return result


class LabelBalancedDataset(Dataset):
    """让每个真实公司名在每个 epoch 贡献相同数量的增强样本。

    原始图片不会复制到磁盘。低频公司通过在线增强重复采样，高频公司的不同图片
    则跨 epoch 轮换，避免少数高频公司支配后训练梯度和最佳模型指标。
    """

    def __init__(
        self,
        dataset: Dataset,
        samples: Sequence[TrainingSample],
        samples_per_label: int,
        seed: int,
    ) -> None:
        if len(dataset) != len(samples):
            raise ValueError("均衡采样的数据集和标签数量不一致")
        if samples_per_label <= 0:
            raise ValueError("samples_per_label 必须大于 0")
        indices_by_label = defaultdict(list)
        for index, sample in enumerate(samples):
            indices_by_label[sample.label].append(index)
        if not indices_by_label:
            raise ValueError("均衡采样的数据集不能为空")

        self.dataset = dataset
        self.samples_per_label = samples_per_label
        self.seed = seed
        self.epoch = 0
        self.labels = sorted(indices_by_label)
        self.indices_by_label: Mapping[str, List[int]] = dict(indices_by_label)

    def __len__(self) -> int:
        return len(self.labels) * self.samples_per_label

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        label_slot = idx % len(self.labels)
        occurrence = idx // len(self.labels)
        label = self.labels[label_slot]
        candidates = self.indices_by_label[label]
        digest = blake2b(
            f"{self.seed}:{self.epoch}:{label}".encode("utf-8"),
            digest_size=8,
        ).digest()
        offset = int.from_bytes(digest, byteorder="big") % len(candidates)
        source_index = candidates[(offset + occurrence) % len(candidates)]
        return self.dataset[source_index]


class ReplayMixDataset(Dataset):
    """
    把真实业务样本与少量合成回放样本组成一个固定比例的训练 epoch。

    真实数据的每张图片在每个 epoch 恰好出现一次。合成部分根据 ``seed``、epoch
    和槽位从完整合成池稳定抽取，因此不会只反复微调同一小撮合成公司名，也不会把
    十万级合成池整体塞进后训练而淹没真实数据。
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        replay_dataset: Dataset,
        replay_ratio: float,
        seed: int,
    ) -> None:
        if len(primary_dataset) <= 0:
            raise ValueError("真实主数据集不能为空")
        if len(replay_dataset) <= 0:
            raise ValueError("合成回放数据集不能为空")
        if not 0 < replay_ratio < 1:
            raise ValueError("replay_ratio 必须在 0 和 1 之间")

        self.primary_dataset = primary_dataset
        self.replay_dataset = replay_dataset
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.epoch = 0
        self.primary_count = len(primary_dataset)
        self.total_count = ceil(self.primary_count / (1 - replay_ratio))
        self.replay_count = self.total_count - self.primary_count

    def __len__(self) -> int:
        return self.total_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))
        if hasattr(self.primary_dataset, "set_epoch"):
            self.primary_dataset.set_epoch(self.epoch)
        if hasattr(self.replay_dataset, "set_epoch"):
            self.replay_dataset.set_epoch(self.epoch)

    def _replay_index(self, slot: int) -> int:
        digest = blake2b(
            f"{self.seed}:{self.epoch}:{slot}".encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, byteorder="big") % len(self.replay_dataset)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_count:
            raise IndexError(idx)
        if idx < self.primary_count:
            return self.primary_dataset[idx]
        replay_slot = idx - self.primary_count
        return self.replay_dataset[self._replay_index(replay_slot)]


def encode_text(
    text,
    max_target_length: int = 128,
    vocab: Optional[Mapping[str, int]] = None,
):
    """按字符编码，并显式加入 BOS/EOS 与定长 PAD。"""
    if vocab is None:
        raise ValueError("vocab 不能为空")
    if not isinstance(text, list):
        text = list(text)

    bos = vocab.get("<s>")
    eos = vocab.get("</s>")
    unk = vocab.get("<unk>")
    pad = vocab.get("<pad>")
    if None in (bos, eos, unk, pad):
        raise ValueError("词表必须包含 <s>、</s>、<unk>、<pad>")

    tokens = [bos]
    tokens.extend(vocab.get(token, unk) for token in text[: max_target_length - 2])
    tokens.append(eos)
    tokens.extend([pad] * (max_target_length - len(tokens)))
    return tokens


def decode_text(tokens, vocab, vocab_inp):
    """把字符 token 解码为公司名；遇到 EOS 后停止。"""
    start = vocab.get("<s>")
    end = vocab.get("</s>")
    unknown = vocab.get("<unk>")
    pad = vocab.get("<pad>")
    ignored = {start, pad, unknown}

    text = []
    for token in tokens:
        token = int(token)
        if token == end:
            break
        if token in ignored:
            continue
        decoded = vocab_inp.get(token)
        if decoded is not None:
            text.append(decoded)
    return "".join(text)

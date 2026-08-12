"""在印章图片与标签对上评估 OCR 模型。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# 屏蔽 VisionEncoderDecoderModel.from_pretrained() 的「encoder/decoder config
# 被共享配置覆盖」提示；image_size/词表等覆盖由 prepare.py 显式写
# 入配置，属预期行为。transformers 对这类覆盖既可能发 UserWarning 也可能
# 走 logging.INFO，下面同时按 warnings 和 logging 做了屏蔽。
warnings.filterwarnings(
    "ignore",
    message=(
        r"Config of the encoder: .* is overwritten by shared encoder config:"
    ),
    category=UserWarning,
    module=r"transformers",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"Config of the decoder: .* is overwritten by shared decoder config:"
    ),
    category=UserWarning,
    module=r"transformers",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"`num_beams` is set to 1\. However, `length_penalty` is set to"
    ),
    category=UserWarning,
    module=r"transformers",
)
# 屏蔽 use_fast 慢处理器迁移提示；本项目使用 prepare.py 生成的自定义
# BPE 词表，v4.52 前后行为对业务没有影响。
warnings.filterwarnings(
    "ignore",
    message=(
        r"Using a slow image processor as `use_fast` is unset and a slow "
        r"processor was saved with this model\."
    ),
    category=UserWarning,
    module=r"transformers",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r"Using a slow processor as `use_fast` is unset and a slow processor "
        r"was saved with this model\."
    ),
    category=UserWarning,
    module=r"transformers",
)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
# 注意：TrOCRProcessor/VisionEncoderDecoderModel 会把 use_fast 迁移提示和
# shared config 覆盖以 logging.INFO/logger.warning 的形式发到不同 logger，
# 单用 warnings.filterwarnings 无法完全覆盖。这里安装一个全局 transformers
# filter 精确屏蔽，其他告警正常输出。
class _TransformersNoiseFilter(logging.Filter):
    """屏蔽已知的、与本项目配置兼容的 transformers 噪声日志。"""

    _NOISE_PREFIXES = (
        "Some weights of the model checkpoint at",
        "This IS expected if you are initializing",
        "This IS NOT expected if you are initializing",
        "Config of the encoder:",
        "Config of the decoder:",
    )
    _NOISE_CONTAINS = (
        "Using a slow image processor as `use_fast` is unset",
        "Using a slow processor as `use_fast` is unset",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if any(message.startswith(p) for p in self._NOISE_PREFIXES):
            return False
        if any(s in message for s in self._NOISE_CONTAINS):
            return False
        return True


_noise_filter = _TransformersNoiseFilter()
for logger_name in (
    "transformers",
    "transformers.modeling_utils",
    "transformers.models.vision_encoder_decoder",
    "transformers.processing_utils",
    "transformers.configuration_utils",
    "transformers.image_utils",
    "transformers.image_processing_utils",
    "transformers.tokenization_utils_base",
    "transformers.generation.configuration_utils",
):
    logging.getLogger(logger_name).addFilter(_noise_filter)
logging.getLogger("transformers.models.vision_encoder_decoder").setLevel(
    logging.WARNING
)
logging.getLogger("transformers.processing_utils").setLevel(logging.WARNING)

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from seal_ocr.dataset import build_spatial_targets, decode_text
from seal_ocr.data import (
    TrainingSample,
    discover_samples,
    normalize_company_name,
)
from seal_ocr.image import (
    prepare_image_for_processor,
    prepare_spatial_annotation_bundle_for_processor,
    processor_image_size,
    resolve_resize_mode,
)
from seal_ocr.length_control import (
    LengthConditionModule,
    LengthControlLogitsProcessor,
    infer_current_step,
)
from seal_ocr.lexicon import CompanyNameTokenTrie, load_company_names
from seal_ocr.spatial_annotations import (
    SPATIAL_CHANNELS,
    load_spatial_annotation_bundle,
    spatial_detail_annotation_path,
)
from seal_ocr.spatial_control import (
    SpatialAuxiliaryHead,
    compute_spatial_objective,
    infer_patch_grid,
)


@dataclass
class EvalResult:
    image_path: str
    label: str
    pred: str
    exact_match: bool
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    quality: str = ""
    predicted_target_length: int = -1
    length_confidence: float = 0.0
    length_constraint_active: bool = False


class EvalDataset(Dataset):
    def __init__(
        self,
        samples: List[TrainingSample],
        processor: TrOCRProcessor,
        force_grayscale: bool = False,
        resize_mode: str = "stretch",
        spatial_target_size: Optional[tuple[int, int]] = None,
    ):
        self.samples = samples
        self.processor = processor
        self.force_grayscale = force_grayscale
        self.resize_mode = resize_mode
        self.spatial_target_size = spatial_target_size
        self.spatial_target_channels = len(SPATIAL_CHANNELS)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as opened_image:
            img = opened_image.convert("RGB")
            original_size = img.size
            if self.force_grayscale:
                gray = ImageOps.grayscale(img)
                img = Image.merge("RGB", (gray, gray, gray))
            img = prepare_image_for_processor(
                img,
                processor=self.processor,
                resize_mode=self.resize_mode,
            )
            pixel_values = self.processor(
                img, return_tensors="pt"
            ).pixel_values.squeeze(0)
        result = {
            "idx": idx,
            "pixel_values": pixel_values,
        }
        if self.spatial_target_size is not None:
            if sample.spatial_annotation_path:
                annotation = load_spatial_annotation_bundle(
                    sample.spatial_annotation_path,
                    require_detail=True,
                    load_detail=True,
                )
                if annotation.size != original_size:
                    raise ValueError(
                        "空间标注与评估图片尺寸不一致: "
                        f"image={sample.image_path} {original_size}, "
                        f"annotation={sample.spatial_annotation_path} "
                        f"{annotation.size}"
                    )
                prepared_annotation = (
                    prepare_spatial_annotation_bundle_for_processor(
                        annotation,
                        processor=self.processor,
                        resize_mode=self.resize_mode,
                    )
                )
                spatial_targets = build_spatial_targets(
                    prepared_annotation,
                    self.spatial_target_size,
                )
                has_spatial_annotation = True
            else:
                spatial_targets = torch.zeros(
                    (
                        self.spatial_target_channels,
                        *self.spatial_target_size,
                    ),
                    dtype=torch.float32,
                )
                has_spatial_annotation = False
            result["spatial_targets"] = spatial_targets
            result["has_spatial_annotation"] = torch.tensor(
                has_spatial_annotation,
                dtype=torch.bool,
            )
        return result


def compute_edit_stats(pred: str, ref: str) -> tuple[float, int, int, int]:
    """返回单条 CER 及替换/删除/插入数，使用训练时的同一算法。"""
    from seal_ocr.metrics import _edit_distance

    S, D, I, C = _edit_distance(list(ref), list(pred))
    total = S + D + C
    cer = (S + D + I) / total if total > 0 else 0.0
    return cer, S, D, I


def find_seal_dataset_samples(
    dataset_dir: str,
    spatial_annotation_path: Optional[str] = None,
) -> List[TrainingSample]:
    """在 seal_dataset 目录中发现所有 (image, label) 配对。"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    samples, issues = discover_samples(
        [str(dataset_path)],
        verify_images=False,
        spatial_annotation_paths=(
            [spatial_annotation_path] if spatial_annotation_path else None
        ),
        auto_spatial_annotations=spatial_annotation_path is None,
        require_spatial_annotations=spatial_annotation_path is not None,
    )
    if issues:
        if spatial_annotation_path is not None:
            examples = [
                f"{issue.issue_type}: {issue.path}"
                for issue in issues[:10]
            ]
            raise ValueError(
                "显式空间标注目录未完整覆盖评估集，拒绝在缺样本子集上计算指标。"
                f"问题数={len(issues)}，示例={examples}"
            )
        print(f"[WARN] 发现 {len(issues)} 条数据问题，跳过", file=sys.stderr)
    if not samples:
        raise ValueError(f"在 {dataset_dir} 下未找到任何图片+标签配对")
    return samples


def find_manifest_samples(
    split_manifest: str,
    manifest_split: str,
) -> List[TrainingSample]:
    """读取 train.py 固定的数据切分，保证不同 beam/模型比较同一批图片。"""
    manifest_path = Path(split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"数据切分文件不存在: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples", {}).get(manifest_split)
    if raw_samples is None:
        available = sorted(payload.get("samples", {}))
        raise ValueError(
            f"数据切分中不存在 {manifest_split!r}，可选值: {available}"
        )

    def resolve_manifest_path(raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return str(path)
        # train.py 当前写入绝对路径；对旧 manifest 则先沿用当前工作目录，
        # 再尝试 manifest 所在目录，避免从 checkpoint 目录启动评估时误报全量
        # 图片不存在。
        if path.exists():
            return str(path.resolve())
        relative_to_manifest = manifest_path.parent / path
        if relative_to_manifest.exists():
            return str(relative_to_manifest.resolve())
        return str(path)

    samples = []
    for index, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise ValueError(
                f"切分文件 {manifest_split!r} 第 {index} 项不是对象"
            )
        image_path = item.get("image_path")
        label = item.get("label")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(
                f"切分文件 {manifest_split!r} 第 {index} 项缺少 image_path"
            )
        if not isinstance(label, str) or not normalize_company_name(label):
            raise ValueError(
                f"切分文件 {manifest_split!r} 第 {index} 项缺少有效 label"
            )
        annotation_path = item.get("spatial_annotation_path")
        if annotation_path is not None and not isinstance(annotation_path, str):
            raise ValueError(
                f"切分文件 {manifest_split!r} 第 {index} 项空间标注路径无效"
            )
        samples.append(
            TrainingSample(
                image_path=resolve_manifest_path(image_path),
                label=normalize_company_name(label),
                spatial_annotation_path=(
                    resolve_manifest_path(annotation_path)
                    if annotation_path
                    else None
                ),
            )
        )
    missing_paths = [
        sample.image_path
        for sample in samples
        if not Path(sample.image_path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"切分文件中有 {len(missing_paths)} 张图片不存在，示例: "
            f"{missing_paths[:5]}"
        )
    if not samples:
        raise ValueError(f"数据切分 {manifest_split!r} 为空")
    image_paths = [sample.image_path for sample in samples]
    duplicate_paths = sorted(
        path
        for path, count in Counter(image_paths).items()
        if count > 1
    )
    if duplicate_paths:
        raise ValueError(
            f"数据切分 {manifest_split!r} 含重复图片路径，示例: "
            f"{duplicate_paths[:5]}"
        )
    return samples


def limit_eval_samples(
    samples: List[TrainingSample],
    max_samples: int,
    seed: int,
) -> List[TrainingSample]:
    selected = list(samples)
    if max_samples <= 0 or len(selected) <= max_samples:
        return selected
    indices = sorted(
        random.Random(seed).sample(range(len(selected)), max_samples)
    )
    return [selected[index] for index in indices]


def length_bucket(length: int) -> str:
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    if length <= 16:
        return "13-16"
    return ">=17"


def summarize_by_length(results: List[EvalResult]) -> Dict[str, dict]:
    buckets: Dict[str, List[EvalResult]] = {}
    for result in results:
        buckets.setdefault(length_bucket(len(result.label)), []).append(result)
    summary = {}
    for name in ("<=8", "9-12", "13-16", ">=17"):
        values = buckets.get(name, [])
        if not values:
            continue
        summary[name] = {
            "count": len(values),
            "exact_match_accuracy": sum(v.exact_match for v in values) / len(values),
            "cer": sum(
                v.substitutions + v.deletions + v.insertions
                for v in values
            )
            / sum(len(v.label) for v in values),
            "macro_average_cer": sum(v.cer for v in values) / len(values),
        }
    return summary


def label_frequency_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    if count <= 100:
        return "21-100"
    return ">100"


def summarize_by_label_frequency(results: List[EvalResult]) -> Dict[str, dict]:
    """同时报告样本微平均和公司名宏平均，避免高频公司支配指标。"""
    label_counts = Counter(result.label for result in results)
    label_values: Dict[str, List[EvalResult]] = defaultdict(list)
    for result in results:
        label_values[result.label].append(result)

    bucket_labels: Dict[str, List[str]] = defaultdict(list)
    for label, count in label_counts.items():
        bucket_labels[label_frequency_bucket(count)].append(label)

    summary = {}
    for name in ("1", "2-5", "6-20", "21-100", ">100"):
        labels = bucket_labels.get(name, [])
        if not labels:
            continue
        values = [result for label in labels for result in label_values[label]]
        label_exact_rates = [
            sum(result.exact_match for result in label_values[label])
            / len(label_values[label])
            for label in labels
        ]
        summary[name] = {
            "sample_count": len(values),
            "unique_label_count": len(labels),
            "exact_match_accuracy": sum(v.exact_match for v in values)
            / len(values),
            "macro_exact_match_accuracy": sum(label_exact_rates)
            / len(label_exact_rates),
            "macro_average_cer": sum(v.cer for v in values) / len(values),
        }
    return summary


def summarize_by_seen_labels(
    results: List[EvalResult],
    seen_labels: set[str],
) -> Dict[str, dict]:
    if not seen_labels:
        return {}
    buckets = {
        "seen_in_real_train": [result for result in results if result.label in seen_labels],
        "unseen_in_real_train": [result for result in results if result.label not in seen_labels],
    }
    summary = {}
    for name, values in buckets.items():
        if not values:
            continue
        reference_characters = sum(len(value.label) for value in values)
        edit_count = sum(
            value.substitutions + value.deletions + value.insertions
            for value in values
        )
        summary[name] = {
            "count": len(values),
            "unique_label_count": len({value.label for value in values}),
            "exact_match_accuracy": sum(value.exact_match for value in values)
            / len(values),
            "cer": edit_count / reference_characters,
        }
    return summary


def load_quality_manifest(manifest_path: str) -> Dict[str, str]:
    """读取 ``image_path,quality`` CSV；以文件名匹配当前评测样本。"""
    path = Path(manifest_path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"image_path", "quality"}.issubset(rows[0]):
        raise ValueError("quality_manifest 必须包含 image_path,quality 两列")
    qualities = {}
    for row in rows:
        image_name = Path(row["image_path"].strip()).name
        quality = row["quality"].strip()
        if not image_name or not quality:
            raise ValueError(f"quality_manifest 有空值: {row}")
        previous = qualities.setdefault(image_name, quality)
        if previous != quality:
            raise ValueError(f"同一图片有冲突质量标签: {image_name}")
    return qualities


def summarize_by_quality(results: List[EvalResult]) -> Dict[str, dict]:
    buckets: Dict[str, List[EvalResult]] = defaultdict(list)
    for result in results:
        if result.quality:
            buckets[result.quality].append(result)
    summary = {}
    for quality, values in sorted(buckets.items()):
        edit_count = sum(
            value.substitutions + value.deletions + value.insertions
            for value in values
        )
        reference_characters = sum(len(value.label) for value in values)
        summary[quality] = {
            "count": len(values),
            "exact_match_accuracy": sum(value.exact_match for value in values)
            / len(values),
            "cer": edit_count / reference_characters,
        }
    return summary


def _detect_length_control_keys(state_dict: dict) -> bool:
    """检测 state_dict 是否包含长度控制模块的参数。"""
    return any(
        k.startswith("length_control.")
        for k in state_dict.keys()
    )


def _detect_spatial_control_keys(state_dict: dict) -> bool:
    """检测 state_dict 是否包含空间辅助头参数。"""
    return any(key.startswith("spatial_head.") for key in state_dict)


def _apply_spatial_control_structure(model, state_dict: dict):
    """按 checkpoint 权重与配置重建固定八通道空间辅助头。"""
    output_key = "spatial_head.refinement.3.weight"
    projection_key = "spatial_head.token_projection.weight"
    if output_key not in state_dict or projection_key not in state_dict:
        return False
    output_channels = int(state_dict[output_key].shape[0])
    if output_channels != len(SPATIAL_CHANNELS):
        raise ValueError(
            "checkpoint 空间头输出通道数必须是 8，"
            f"实际为 {output_channels}"
        )
    head_hidden_size = int(state_dict[projection_key].shape[0])
    encoder_hidden_size = int(state_dict[projection_key].shape[1])
    saved_config = getattr(model.config, "spatial_control_config", {}) or {}
    configured_channels = int(
        saved_config.get("output_channels", output_channels)
    )
    if configured_channels != output_channels:
        raise ValueError(
            "checkpoint 空间头权重与 config 通道数不一致: "
            f"weights={output_channels}, config={configured_channels}"
        )
    architecture_version = int(saved_config.get("architecture_version", 0))
    if architecture_version != 2:
        raise ValueError(
            "checkpoint 空间头 architecture_version 必须是 2，"
            f"实际为 {architecture_version}"
        )
    raw_grid_size = saved_config.get("grid_size")
    if raw_grid_size is None:
        raise ValueError("checkpoint 空间头 config 缺少 grid_size")
    grid_size = tuple(int(value) for value in raw_grid_size)

    model.spatial_head = SpatialAuxiliaryHead(
        encoder_hidden_size=encoder_hidden_size,
        head_hidden_size=head_hidden_size,
        grid_size=grid_size,
    )
    print(
        "[INFO] 重建空间辅助头: "
        f"channels={output_channels}, grid={grid_size}, "
        f"head_hidden={head_hidden_size}"
    )
    return True


def _apply_length_control_structure(model, state_dict: dict):
    """从 state_dict 重建长度控制模块。

    新架构不再替换 decoder embed_tokens，length_control 作为独立模块
    挂载到 model 上。长度条件通过 forward 包装在 logits 输出层注入。
    """
    # 收集 length_control 参数
    lc_keys = [k for k in state_dict.keys() if k.startswith("length_control.")]
    if not lc_keys:
        return False

    # 从 state_dict 推断 length_control 配置
    # length_control.eos_bias_embedding.weight shape: (2*max+1, 1)
    eos_bias_key = "length_control.eos_bias_embedding.weight"
    if eos_bias_key not in state_dict:
        return False
    eos_bias_shape = state_dict[eos_bias_key].shape
    max_target_length = (eos_bias_shape[0] - 1) // 2

    # length_control.hidden_bias_embedding.weight shape: (2*max+1, decoder_hidden)
    hidden_bias_key = "length_control.hidden_bias_embedding.weight"
    if hidden_bias_key not in state_dict:
        return False
    decoder_hidden_size = state_dict[hidden_bias_key].shape[1]

    # length_control.length_predictor.0.weight shape: (predictor_hidden, encoder_hidden)
    predictor_key = "length_control.length_predictor.0.weight"
    if predictor_key not in state_dict:
        return False
    encoder_hidden_size = state_dict[predictor_key].shape[1]
    predictor_hidden_size = state_dict[predictor_key].shape[0]
    saved_config = getattr(model.config, "length_control_config", {}) or {}
    architecture_version = int(saved_config.get("architecture_version", 0))
    if architecture_version != 3:
        raise ValueError(
            "checkpoint 长度头 architecture_version 必须是 3，"
            f"实际为 {architecture_version}"
        )
    if saved_config.get("conditioning_mode") != "predicted_detached":
        raise ValueError("checkpoint 长度头 conditioning_mode 必须是 predicted_detached")
    configured_max_length = int(
        saved_config.get("max_target_length", max_target_length)
    )
    if configured_max_length != max_target_length:
        raise ValueError(
            "长度头权重与 config 的 max_target_length 不一致: "
            f"weights={max_target_length}, config={configured_max_length}"
        )
    predictor_dropout = float(saved_config.get("predictor_dropout", 0.1))

    # 构建 LengthConditionModule
    length_control = LengthConditionModule(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        max_target_length=max_target_length,
        predictor_hidden_size=predictor_hidden_size,
        predictor_dropout=predictor_dropout,
    )
    model.length_control = length_control

    print(
        f"[INFO] 重建长度控制模块: max_target_length={max_target_length}, "
        f"encoder_hidden={encoder_hidden_size}, decoder_hidden={decoder_hidden_size}, "
        f"predictor_hidden={predictor_hidden_size}"
    )
    return True


def _load_single_checkpoint_state(checkpoint_dir: Path):
    """读取单文件 checkpoint，并清理标准 DDP 的 ``module.`` 前缀。"""
    safe_file = checkpoint_dir / "model.safetensors"
    bin_file = checkpoint_dir / "pytorch_model.bin"
    if safe_file.exists():
        from safetensors.torch import load_file as safe_load_file

        state_dict = safe_load_file(str(safe_file), device="cpu")
    elif bin_file.exists():
        state_dict = torch.load(
            str(bin_file),
            map_location="cpu",
            weights_only=True,
        )
    else:
        return None
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model(
    cust_data_init_weights_path: str,
    checkpoint: Optional[str],
):
    """加载 processor 和 model；若指定 checkpoint，从 checkpoint 加载权重。

    支持长度控制模块：检测 checkpoint 是否包含 length_control 参数，
    如果包含则重建 LengthConditionModule 并加载权重。
    长度条件通过 _wrap_forward_for_length_condition 在推理时注入。
    """
    processor = TrOCRProcessor.from_pretrained(
        cust_data_init_weights_path, trust_remote_code=True
    )
    model = None
    loaded_from = cust_data_init_weights_path
    raw_state_dict = None

    if checkpoint and Path(checkpoint).exists():
        ckpt_path = Path(checkpoint)
        safe_file = ckpt_path / "model.safetensors"
        bin_file = ckpt_path / "pytorch_model.bin"
        config_file = ckpt_path / "config.json"

        # 先加载 state_dict 用于检测长度控制结构
        raw_state_dict = _load_single_checkpoint_state(ckpt_path)

        has_length_control = (
            raw_state_dict is not None
            and _detect_length_control_keys(raw_state_dict)
        )
        has_spatial_control = (
            raw_state_dict is not None
            and _detect_spatial_control_keys(raw_state_dict)
        )

        if config_file.exists() and (safe_file.exists() or bin_file.exists()):
            model = VisionEncoderDecoderModel.from_pretrained(str(ckpt_path))
            processor_files = (
                ckpt_path / "preprocessor_config.json",
                ckpt_path / "tokenizer_config.json",
                ckpt_path / "tokenizer.json",
            )
            if all(path.exists() for path in processor_files):
                processor = TrOCRProcessor.from_pretrained(
                    str(ckpt_path), trust_remote_code=True
                )
            loaded_from = str(ckpt_path)
            print(f"[INFO] 从完整 checkpoint 加载模型与生成配置: {ckpt_path}")

            # 标准模型类不会自行创建自定义辅助模块，先重建再统一加载权重。
            if (
                raw_state_dict is not None
                and (has_length_control or has_spatial_control)
            ):
                if has_length_control:
                    _apply_length_control_structure(model, raw_state_dict)
                if has_spatial_control:
                    _apply_spatial_control_structure(model, raw_state_dict)
                # 重建后结构与 checkpoint 一致，直接加载
                missing, unexpected = model.load_state_dict(
                    raw_state_dict, strict=False
                )
                if missing:
                    print(
                        f"[WARN] 重建后仍缺失 {len(missing)} 个 key，示例: {missing[:5]}"
                    )
                if unexpected:
                    print(
                        f"[WARN] 重建后多出 {len(unexpected)} 个 key，示例: {unexpected[:5]}"
                    )
        elif raw_state_dict is not None:
            model = VisionEncoderDecoderModel.from_pretrained(
                cust_data_init_weights_path
            )
            if has_length_control:
                _apply_length_control_structure(model, raw_state_dict)
            if has_spatial_control:
                _apply_spatial_control_structure(model, raw_state_dict)
            missing, unexpected = model.load_state_dict(
                raw_state_dict, strict=False
            )
            loaded_from = str(ckpt_path)
            if missing:
                print(
                    f"[WARN] checkpoint 缺失 {len(missing)} 个 key，示例: {missing[:5]}"
                )
            if unexpected:
                print(
                    f"[WARN] checkpoint 多出 {len(unexpected)} 个 key，示例: {unexpected[:5]}"
                )
            print(f"[INFO] 从 checkpoint 加载权重: {ckpt_path}")
        else:
            print(
                f"[WARN] checkpoint 目录下未找到 model.safetensors 或 "
                f"pytorch_model.bin，回退到基线权重"
            )
    else:
        print("[INFO] 从 cust_data_init_weights_path 加载基线模型权重")

    if model is None:
        model = VisionEncoderDecoderModel.from_pretrained(
            cust_data_init_weights_path
        )
        raw_state_dict = _load_single_checkpoint_state(
            Path(cust_data_init_weights_path)
        )
        if raw_state_dict is not None:
            has_length_control = _detect_length_control_keys(raw_state_dict)
            has_spatial_control = _detect_spatial_control_keys(raw_state_dict)
            if has_length_control:
                _apply_length_control_structure(model, raw_state_dict)
            if has_spatial_control:
                _apply_spatial_control_structure(model, raw_state_dict)
            if has_length_control or has_spatial_control:
                missing, unexpected = model.load_state_dict(
                    raw_state_dict,
                    strict=False,
                )
                if missing:
                    print(
                        f"[WARN] 基线权重缺失 {len(missing)} 个 key，"
                        f"示例: {missing[:5]}"
                    )
                if unexpected:
                    print(
                        f"[WARN] 基线权重多出 {len(unexpected)} 个 key，"
                        f"示例: {unexpected[:5]}"
                    )

    print(f"[INFO] 最终权重来源: {loaded_from}")
    return processor, model


def _wrap_forward_for_length_condition(
    model,
    eos_token_id: int,
    condition_scale: float = 1.0,
):
    """包装 model.forward，在推理时注入长度条件。

    长度条件通过 apply_length_condition 在 logits 输出层注入：
    - EOS bias：直接影响 EOS logit
    - hidden bias：影响隐藏状态（从而影响所有 token 预测）

    推理时完整使用 stop-gradient 的预测长度分布，与训练文字支路一致。
    """
    length_control = getattr(model, "length_control", None)
    if length_control is None:
        return

    original_forward = model.forward
    output_projection = model.get_output_embeddings()
    if output_projection is None:
        raise RuntimeError("decoder 未提供输出投影，无法注入长度条件")

    def _forward_with_length_condition(*args, **kwargs):
        kwargs.setdefault("output_hidden_states", True)
        kwargs["return_dict"] = True

        # 获取或计算 encoder 输出
        encoder_outputs = kwargs.get("encoder_outputs")
        if encoder_outputs is None:
            pixel_values = kwargs.get("pixel_values")
            if pixel_values is None and args:
                pixel_values = args[0]
            if pixel_values is not None:
                encoder_outputs = model.encoder(pixel_values=pixel_values)
                kwargs["encoder_outputs"] = encoder_outputs

        # 计算长度 logits
        length_logits = None
        if encoder_outputs is not None:
            encoder_hidden = encoder_outputs.last_hidden_state
            with torch.no_grad():
                length_logits = length_control(encoder_hidden.detach())

        # 调用原始 forward
        outputs = original_forward(*args, **kwargs)

        # 在 logits 输出层添加长度条件
        if length_logits is not None and outputs.logits is not None:
            hidden_states = None
            if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
                hidden_states = outputs.decoder_hidden_states[-1]
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]

            if hidden_states is not None:
                current_step = infer_current_step(kwargs)
                new_logits, _ = length_control.apply_length_condition(
                    logits=outputs.logits,
                    hidden_states=hidden_states,
                    length_logits=length_logits,
                    eos_token_id=eos_token_id,
                    output_projection=output_projection,
                    current_step=current_step,
                    condition_scale=condition_scale,
                )
                outputs.logits = new_logits

        return outputs

    model.forward = _forward_with_length_condition


def main():
    parser = argparse.ArgumentParser(
        description="在 seal_dataset 上评估 OCR 完全匹配准确率"
    )
    parser.add_argument(
        "--cust_data_init_weights_path",
        default="./models/init",
        type=str,
        help="自定义词表模型目录（含 processor/config/词表）",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./seal_dataset",
        type=str,
        help="待评估数据目录；指定 --split_manifest 时忽略",
    )
    parser.add_argument(
        "--spatial_annotation_path",
        default=None,
        type=str,
        help=(
            "可选空间标注根目录；未指定时自动探测 dataset_dir 同级的"
            " <dataset>_spatial。split_manifest 会直接读取其中保存的标注路径"
        ),
    )
    parser.add_argument(
        "--split_manifest",
        default=None,
        type=str,
        help="train.py 写出的 data_split.json，用于复现完全相同的验证切分",
    )
    parser.add_argument(
        "--manifest_split",
        choices=["train", "eval", "test", "replay"],
        default="eval",
        help="从 data_split.json 读取哪个集合",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="训练后 checkpoint 路径（如 checkpoint-200 目录）；为空则用基线权重",
    )
    parser.add_argument(
        "--CUDA_VISIBLE_DEVICES", default="0", type=str, help="GPU 编号"
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="推理 batch size"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="DataLoader 进程数"
    )
    parser.add_argument(
        "--max_new_tokens",
        default=40,
        type=int,
        help="generate 最大 token 数（不含 BOS）",
    )
    parser.add_argument(
        "--min_new_tokens",
        default=0,
        type=int,
        help="通用最少生成 token 数；长度控制模型建议保持 0",
    )
    parser.add_argument(
        "--length_tolerance",
        default=None,
        type=int,
        help="高置信长度的 EOS 容差；默认读取 checkpoint，未保存时为 0",
    )
    parser.add_argument(
        "--length_force_confidence",
        default=None,
        type=float,
        help="启用精确 EOS 约束的最低长度置信度；默认读取 checkpoint",
    )
    parser.add_argument(
        "--length_condition_scale",
        default=1.0,
        type=float,
        help=(
            "预测长度软条件强度，范围 [0,1]；最终部署默认 1。"
            "评估激活期 checkpoint 时可传训练日志中的 length_condition_scale"
        ),
    )
    parser.add_argument(
        "--no_length_hard_constraint",
        action="store_true",
        help="保留长度头和软条件，但关闭强制 EOS；用于判断指标下降是否来自硬截断",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="生成 beam 数；建议用 1 和 4 对同一 split 各评估一次",
    )
    parser.add_argument(
        "--length_penalty",
        default=1.0,
        type=float,
        help="beam search 长度惩罚；值越大越偏向长输出（默认: 1.0）",
    )
    parser.add_argument(
        "--beam_early_stopping",
        action="store_true",
        help="显式启用 beam early_stopping；默认关闭以免偏向过短公司名",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        default=0,
        type=int,
        help="禁止重复 n-gram 的长度；公司名识别默认关闭（0）",
    )
    parser.add_argument(
        "--max_samples",
        default=0,
        type=int,
        help="确定性抽样上限；0 表示评估全部样本",
    )
    parser.add_argument(
        "--force_grayscale",
        action="store_true",
        help="把全部输入转为三通道灰度，用于检验模型是否依赖印章颜色",
    )
    parser.add_argument(
        "--resize_mode",
        choices=["auto", "stretch", "letterbox"],
        default="auto",
        help="auto 读取模型训练时保存的模式；letterbox 等比例补边",
    )
    parser.add_argument(
        "--lexicon_path",
        nargs="+",
        default=None,
        help=(
            "可选平台注册公司名单；指定后使用字符 trie 约束解码。"
            "只适用于业务确认真实公司必在名单内的指标"
        ),
    )
    seen_label_group = parser.add_mutually_exclusive_group()
    seen_label_group.add_argument(
        "--seen_label_dataset_path",
        nargs="+",
        default=None,
        help="可选真实训练目录，用于把测试指标拆成真实训练已见/未见公司",
    )
    parser.add_argument(
        "--quality_manifest",
        default=None,
        help="可选 image_path,quality CSV，用于单列 normal/hard/unreadable 指标",
    )
    seen_label_group.add_argument(
        "--seen_label_split_manifest",
        default=None,
        help="推荐：finetune 的 data_split.json，以其中 train 标签判定已见公司",
    )
    parser.add_argument("--seed", default=10086, type=int)
    parser.add_argument(
        "--report_csv",
        default=None,
        type=str,
        help="导出每条结果到 CSV（默认: seal_dataset_eval_report.csv）",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        type=str,
        help="导出聚合统计到 JSON（默认: seal_dataset_eval_summary.json）",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if args.num_workers < 0:
        raise ValueError("num_workers 必须大于等于 0")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")
    if args.num_beams <= 0:
        raise ValueError("num_beams 必须大于 0")
    if args.min_new_tokens < 0:
        raise ValueError("min_new_tokens 必须大于等于 0")
    if args.min_new_tokens >= args.max_new_tokens:
        raise ValueError("min_new_tokens 必须小于 max_new_tokens")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size 必须大于等于 0")
    if args.max_samples < 0:
        raise ValueError("max_samples 必须大于等于 0")
    if args.length_tolerance is not None and args.length_tolerance < 0:
        raise ValueError("length_tolerance 必须大于等于 0")
    if (
        args.length_force_confidence is not None
        and not 0.0 <= args.length_force_confidence <= 1.0
    ):
        raise ValueError("length_force_confidence 必须在 [0, 1] 范围内")
    if not 0.0 <= args.length_condition_scale <= 1.0:
        raise ValueError("length_condition_scale 必须在 [0, 1] 范围内")
    if args.split_manifest and args.spatial_annotation_path:
        raise ValueError(
            "使用 split_manifest 时空间标注路径已记录在 manifest 中，"
            "不能再指定 spatial_annotation_path"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 发现样本
    if args.split_manifest:
        samples = find_manifest_samples(
            args.split_manifest,
            args.manifest_split,
        )
        dataset_source = f"{args.split_manifest}:{args.manifest_split}"
    else:
        samples = find_seal_dataset_samples(
            args.dataset_dir,
            spatial_annotation_path=args.spatial_annotation_path,
        )
        dataset_source = args.dataset_dir
    original_sample_count = len(samples)
    samples = limit_eval_samples(
        samples,
        args.max_samples,
        args.seed,
    )
    print(f"[INFO] 共发现 {len(samples)} 条标注样本")
    if len(samples) != original_sample_count:
        print(
            f"[INFO] 已按 seed={args.seed} 从 {original_sample_count} 条中"
            f"确定性抽取 {len(samples)} 条"
        )
    seen_labels = set()
    if args.seen_label_dataset_path:
        seen_samples, seen_issues = discover_samples(
            args.seen_label_dataset_path,
            verify_images=False,
        )
        if seen_issues:
            print(
                f"[WARN] seen_label_dataset 有 {len(seen_issues)} 个数据问题，"
                "已使用其中有效标签",
                file=sys.stderr,
            )
        seen_labels = {sample.label for sample in seen_samples}
    elif args.seen_label_split_manifest:
        seen_labels = {
            sample.label
            for sample in find_manifest_samples(
                args.seen_label_split_manifest,
                "train",
            )
        }
    quality_labels = (
        load_quality_manifest(args.quality_manifest)
        if args.quality_manifest
        else {}
    )
    if quality_labels:
        missing_quality = [
            Path(sample.image_path).name
            for sample in samples
            if Path(sample.image_path).name not in quality_labels
        ]
        if missing_quality:
            raise ValueError(
                "quality_manifest 未覆盖全部评测图片，示例: "
                f"{missing_quality[:10]}"
            )

    # 2. 加载模型
    processor, model = load_model(
        args.cust_data_init_weights_path, args.checkpoint
    )
    resize_mode = resolve_resize_mode(args.resize_mode, model)
    print(f"[INFO] 图片尺寸模式: {resize_mode}")
    model.to(device)
    model.eval()

    saved_spatial_config = (
        getattr(model.config, "spatial_control_config", {}) or {}
    )
    spatial_head = getattr(model, "spatial_head", None)
    if saved_spatial_config.get("enabled", False) and spatial_head is None:
        raise RuntimeError(
            "checkpoint config 声明启用了空间头，但权重加载后模型中没有 spatial_head"
        )
    spatial_grid_size = None
    spatial_target_channels = len(SPATIAL_CHANNELS)
    spatial_objective_weights = {
        "text_weight": float(saved_spatial_config.get("text_weight", 1.0)),
        "stamp_weight": float(saved_spatial_config.get("stamp_weight", 0.5)),
        "heatmap_weight": float(
            saved_spatial_config.get("heatmap_weight", 0.25)
        ),
        "character_weight": float(
            saved_spatial_config.get("character_weight", 1.0)
        ),
    }
    if spatial_head is not None:
        spatial_grid_size = tuple(int(value) for value in spatial_head.grid_size)
        if int(spatial_head.output_channels) != spatial_target_channels:
            raise ValueError("空间头不是当前固定八通道结构")
        target_width, target_height = processor_image_size(processor)
        expected_grid_size = infer_patch_grid(
            model.config.encoder,
            (target_height, target_width),
        )
        if spatial_grid_size != expected_grid_size:
            raise ValueError(
                "空间头网格与当前 processor 不一致: "
                f"head={spatial_grid_size}, processor={expected_grid_size}"
            )
        annotated_count = sum(
            sample.spatial_annotation_path is not None for sample in samples
        )
        missing_detail = [
            str(spatial_detail_annotation_path(sample.spatial_annotation_path))
            for sample in samples
            if sample.spatial_annotation_path is not None
            and not spatial_detail_annotation_path(
                sample.spatial_annotation_path
            ).is_file()
        ]
        if missing_detail:
            raise FileNotFoundError(
                "空间头评估缺少 .detail.png 标注: "
                f"共 {len(missing_detail)} 张，示例 {missing_detail[:5]}"
            )
        print(
            "[INFO] 空间头评估: "
            f"channels={spatial_target_channels}, grid={spatial_grid_size}, "
            f"有标注样本={annotated_count}/{len(samples)}"
        )
        if annotated_count == 0:
            print(
                "[WARN] 当前评估集没有空间标注；仍评估 OCR，但不会产生空间指标。"
                "真实人工集通常属于这种情况，空间头请在固定的合成验证集上评估。",
                file=sys.stderr,
            )

    saved_length_config = getattr(model.config, "length_control_config", {}) or {}
    saved_length_architecture = int(
        saved_length_config.get("architecture_version", 0)
    )
    if saved_length_config and saved_length_architecture != 3:
        raise ValueError(
            "checkpoint 长度头 architecture_version 必须是 3，"
            f"实际为 {saved_length_architecture}"
        )
    length_tolerance = (
        args.length_tolerance
        if args.length_tolerance is not None
        else int(saved_length_config.get("tolerance", 0))
    )
    saved_force_confidence = float(
        saved_length_config.get("force_confidence", 0.90)
    )
    length_force_confidence = (
        args.length_force_confidence
        if args.length_force_confidence is not None
        else saved_force_confidence
    )
    if length_tolerance < 0:
        raise ValueError("checkpoint 中的 length_tolerance 必须大于等于 0")
    if not 0.0 <= length_force_confidence <= 1.0:
        raise ValueError(
            "checkpoint 中的 length_force_confidence 必须在 [0, 1] 范围内"
        )

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {v: k for k, v in vocab.items()}
    lexicon_trie = None
    lexicon_names = set()
    if args.lexicon_path:
        lexicon_names = set(load_company_names(args.lexicon_path))
        lexicon_trie = CompanyNameTokenTrie(
            names=lexicon_names,
            vocab=vocab,
            decoder_start_token_id=model.config.decoder_start_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        print(
            "[INFO] 公司词典约束解码: "
            f"可用 {len(lexicon_trie.accepted_names)}，"
            f"词表字符不全而排除 {len(lexicon_trie.rejected_names)}"
        )
        if args.num_beams == 1:
            print("[WARN] 词典约束建议 num_beams=4，以免首字符局部最优选错公司")

    # 3. DataLoader
    dataset = EvalDataset(
        samples,
        processor,
        force_grayscale=args.force_grayscale,
        resize_mode=resize_mode,
        spatial_target_size=spatial_grid_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    # 4. 推理
    # 如果模型包含长度控制模块，包装 forward 注入长度条件
    length_control = getattr(model, "length_control", None)
    if length_control is not None:
        _wrap_forward_for_length_condition(
            model,
            eos_token_id=processor.tokenizer.eos_token_id,
            condition_scale=args.length_condition_scale,
        )
        print(
            "[INFO] 已启用长度条件注入与高置信 EOS 约束: "
            f"condition_scale={args.length_condition_scale:g}, "
            f"tolerance={length_tolerance}, "
            f"confidence>={length_force_confidence:g}, "
            f"hard_constraint={not args.no_length_hard_constraint}"
        )
        if lexicon_trie is not None:
            print(
                "[INFO] 词典模式下关闭硬长度 EOS 约束，避免预测长度与词典候选冲突；"
                "软长度条件和长度头指标仍保留"
            )

    results: List[EvalResult] = []
    spatial_metric_sums: Dict[str, float] = defaultdict(float)
    spatial_metric_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            idxs = batch["idx"].tolist()
            pixel_values = batch["pixel_values"].to(device)
            has_spatial_annotation = batch.get("has_spatial_annotation")
            if has_spatial_annotation is not None:
                has_spatial_annotation = has_spatial_annotation.to(device).bool()
            needs_spatial_encoder = bool(
                spatial_head is not None
                and has_spatial_annotation is not None
                and has_spatial_annotation.any()
            )
            encoder_outputs = None
            if length_control is not None or needs_spatial_encoder:
                encoder_outputs = model.encoder(pixel_values=pixel_values)

            if needs_spatial_encoder:
                spatial_targets = batch["spatial_targets"].to(device)
                annotated_hidden = encoder_outputs.last_hidden_state[
                    has_spatial_annotation
                ]
                annotated_targets = spatial_targets[has_spatial_annotation]
                spatial_logits = spatial_head(annotated_hidden)
                _, batch_spatial_metrics = compute_spatial_objective(
                    spatial_logits,
                    annotated_targets,
                    **spatial_objective_weights,
                )
                annotated_batch_count = int(
                    has_spatial_annotation.sum().item()
                )
                spatial_metric_count += annotated_batch_count
                for name, value in batch_spatial_metrics.items():
                    spatial_metric_sums[name] += (
                        float(value.detach().cpu().item())
                        * annotated_batch_count
                    )

            batch_predicted_lengths = [-1] * len(idxs)
            batch_length_confidences = [0.0] * len(idxs)
            batch_constraint_active = [False] * len(idxs)

            # 长度控制：预计算离散目标长度，用于指标和高置信 EOS 约束。
            if length_control is not None:
                length_logits = length_control(
                    encoder_outputs.last_hidden_state.detach()
                )
                predicted_length, length_confidence = length_control.decode(
                    length_logits
                )
                effective_length_confidence = (
                    length_confidence * args.length_condition_scale
                )
                batch_predicted_lengths = predicted_length.cpu().tolist()
                batch_length_confidences = length_confidence.cpu().tolist()
                batch_constraint_active = [
                    lexicon_trie is None
                    and not args.no_length_hard_constraint
                    and confidence >= length_force_confidence
                    for confidence in effective_length_confidence.cpu().tolist()
                ]
                logits_processor = None
                if lexicon_trie is None and not args.no_length_hard_constraint:
                    logits_processor = [
                        LengthControlLogitsProcessor(
                            predicted_length=predicted_length,
                            confidence=effective_length_confidence,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            tolerance=length_tolerance,
                            minimum_confidence=length_force_confidence,
                            ignored_token_ids=(
                                model.config.decoder_start_token_id,
                                processor.tokenizer.cls_token_id,
                                processor.tokenizer.pad_token_id,
                                processor.tokenizer.unk_token_id,
                            ),
                        )
                    ]
                # 不传 encoder_outputs：generate 内部会自行调用 encoder，
                # forward 包装会从 generate 传入的 encoder_outputs 计算
                # length_logits 并应用长度条件。
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    early_stopping=args.beam_early_stopping,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    logits_processor=logits_processor,
                    prefix_allowed_tokens_fn=(
                        lexicon_trie.allowed_tokens if lexicon_trie else None
                    ),
                )
            else:
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    early_stopping=args.beam_early_stopping,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    prefix_allowed_tokens_fn=(
                        lexicon_trie.allowed_tokens if lexicon_trie else None
                    ),
                )
            if not torch.is_tensor(generated_ids):
                raise RuntimeError(
                    "model.generate 返回了非 Tensor；请确认未启用 "
                    "return_dict_in_generate"
                )
            if generated_ids.ndim != 2 or generated_ids.size(0) != len(idxs):
                raise RuntimeError(
                    "生成结果与输入 batch 数量不一致: "
                    f"generated_shape={tuple(generated_ids.shape)}, "
                    f"input_batch={len(idxs)}"
                )
            for batch_index, (i, gids) in enumerate(
                zip(idxs, generated_ids.cpu().numpy())
            ):
                sample = samples[i]
                label = normalize_company_name(sample.label)
                pred = decode_text(gids, vocab, vocab_inp)
                pred = normalize_company_name(pred)
                exact = label == pred
                cer, substitutions, deletions, insertions = compute_edit_stats(
                    pred,
                    label,
                )
                results.append(
                    EvalResult(
                        image_path=sample.image_path,
                        label=label,
                        pred=pred,
                        exact_match=exact,
                        cer=cer,
                        substitutions=substitutions,
                        deletions=deletions,
                        insertions=insertions,
                        quality=quality_labels.get(
                            Path(sample.image_path).name,
                            "",
                        ),
                        predicted_target_length=int(
                            batch_predicted_lengths[batch_index]
                        ),
                        length_confidence=float(
                            batch_length_confidences[batch_index]
                        ),
                        length_constraint_active=bool(
                            batch_constraint_active[batch_index]
                        ),
                    )
                )

    # 5. 统计
    n = len(results)
    exact_count = sum(1 for r in results if r.exact_match)
    exact_acc = exact_count / n
    macro_avg_cer = sum(r.cer for r in results) / n
    total_substitutions = sum(r.substitutions for r in results)
    total_deletions = sum(r.deletions for r in results)
    total_insertions = sum(r.insertions for r in results)
    total_reference_characters = sum(len(r.label) for r in results)
    cer = (
        total_substitutions + total_deletions + total_insertions
    ) / total_reference_characters
    length_summary = summarize_by_length(results)
    frequency_summary = summarize_by_label_frequency(results)
    seen_label_summary = summarize_by_seen_labels(results, seen_labels)
    quality_summary = summarize_by_quality(results)

    results_by_label: Dict[str, List[EvalResult]] = defaultdict(list)
    for result in results:
        results_by_label[result.label].append(result)
    per_label_exact_rates = [
        sum(result.exact_match for result in values) / len(values)
        for values in results_by_label.values()
    ]
    macro_exact_match = sum(per_label_exact_rates) / len(per_label_exact_rates)
    zero_exact_label_count = sum(rate == 0 for rate in per_label_exact_rates)
    prediction_counts = Counter(result.pred for result in results)
    top_predictions = [
        {
            "prediction": prediction,
            "count": count,
            "share": count / n,
        }
        for prediction, count in prediction_counts.most_common(20)
    ]
    mean_label_length = sum(len(r.label) for r in results) / n
    mean_prediction_length = sum(len(r.pred) for r in results) / n
    output_length_errors = [len(r.pred) - len(r.label) for r in results]
    output_length_metrics = {
        "accuracy": sum(error == 0 for error in output_length_errors) / n,
        "mae": sum(abs(error) for error in output_length_errors) / n,
        "within_1": sum(abs(error) <= 1 for error in output_length_errors) / n,
        "under_rate": sum(error < 0 for error in output_length_errors) / n,
        "over_rate": sum(error > 0 for error in output_length_errors) / n,
    }
    length_head_results = [
        result for result in results if result.predicted_target_length >= 0
    ]
    length_head_metrics = None
    if length_head_results:
        head_errors = [
            result.predicted_target_length - len(result.label)
            for result in length_head_results
        ]
        head_count = len(head_errors)
        constraint_active_count = sum(
            result.length_constraint_active for result in length_head_results
        )
        constraint_active_correct_count = sum(
            result.length_constraint_active and error == 0
            for result, error in zip(length_head_results, head_errors)
        )
        length_head_metrics = {
            "count": head_count,
            "accuracy": sum(error == 0 for error in head_errors) / head_count,
            "mae": sum(abs(error) for error in head_errors) / head_count,
            "within_1": sum(abs(error) <= 1 for error in head_errors) / head_count,
            "under_rate": sum(error < 0 for error in head_errors) / head_count,
            "over_rate": sum(error > 0 for error in head_errors) / head_count,
            "mean_confidence": sum(
                result.length_confidence for result in length_head_results
            )
            / head_count,
            "constraint_active_count": constraint_active_count,
            "constraint_active_rate": constraint_active_count / head_count,
            "constraint_active_correct_count": constraint_active_correct_count,
            "constraint_active_length_accuracy": (
                constraint_active_correct_count / constraint_active_count
                if constraint_active_count
                else None
            ),
        }
    spatial_head_metrics = None
    if spatial_head is not None:
        spatial_head_metrics = {
            "architecture_version": 2,
            "channel_count": spatial_target_channels,
            "annotated_count": spatial_metric_count,
            "annotation_rate": spatial_metric_count / n,
            "metrics": (
                {
                    name: total / spatial_metric_count
                    for name, total in sorted(spatial_metric_sums.items())
                }
                if spatial_metric_count
                else {}
            ),
        }
    empty_prediction_count = sum(not r.pred for r in results)
    lexicon_covered_count = sum(r.label in lexicon_names for r in results)

    # 按 CER 降序排错误样例 Top-K
    wrongs = [r for r in results if not r.exact_match]
    wrongs_sorted = sorted(wrongs, key=lambda r: r.cer, reverse=True)
    severe_wrong_count = sum(r.cer > 0.5 for r in wrongs)
    very_severe_wrong_count = sum(r.cer > 1.0 for r in wrongs)
    long_overrun_count = sum(
        len(r.pred) - len(r.label) >= 5 for r in wrongs
    )

    print("\n" + "=" * 60)
    print(f"样本总数        : {n}")
    print(f"完全匹配        : {exact_count}")
    print(f"完全匹配准确率  : {exact_acc * 100:.2f}%")
    print(f"公司名宏平均准确率: {macro_exact_match * 100:.2f}%")
    print(f"CER（全局字符） : {cer * 100:.2f}%")
    print(f"CER（样本宏平均）: {macro_avg_cer * 100:.2f}%")
    print(
        f"唯一公司/零命中 : {len(results_by_label)} / "
        f"{zero_exact_label_count}"
    )
    print(
        f"唯一预测/平均长度: {len(prediction_counts)} / "
        f"标签 {mean_label_length:.2f} / 预测 {mean_prediction_length:.2f}"
    )
    print(
        "输出长度        : "
        f"exact {output_length_metrics['accuracy'] * 100:.2f}% / "
        f"MAE {output_length_metrics['mae']:.3f} / "
        f"±1 {output_length_metrics['within_1'] * 100:.2f}% / "
        f"短 {output_length_metrics['under_rate'] * 100:.2f}% / "
        f"长 {output_length_metrics['over_rate'] * 100:.2f}%"
    )
    if length_head_metrics:
        print(
            "长度头          : "
            f"exact {length_head_metrics['accuracy'] * 100:.2f}% / "
            f"MAE {length_head_metrics['mae']:.3f} / "
            f"±1 {length_head_metrics['within_1'] * 100:.2f}% / "
            f"平均置信 {length_head_metrics['mean_confidence']:.3f} / "
            f"硬约束 {length_head_metrics['constraint_active_rate'] * 100:.2f}%"
        )
    if spatial_head_metrics:
        print(
            "空间头          : "
            f"v{spatial_head_metrics['architecture_version']} / "
            f"{spatial_head_metrics['channel_count']} 通道 / "
            f"有标注 {spatial_head_metrics['annotated_count']}/{n}"
        )
        if spatial_head_metrics["metrics"]:
            print(
                "空间头指标      : "
                + json.dumps(
                    spatial_head_metrics["metrics"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    print(f"错误样本数      : {len(wrongs)} ({len(wrongs) / n * 100:.2f}%)")
    print(
        "字符错误构成    : "
        f"替换 {total_substitutions} / 删除 {total_deletions} / "
        f"插入 {total_insertions}"
    )
    print(
        "严重错误        : "
        f"CER>50% {severe_wrong_count} / CER>100% {very_severe_wrong_count} / "
        f"预测多出>=5字 {long_overrun_count}"
    )
    print(f"beam 数         : {args.num_beams}")
    print(f"长度惩罚        : {args.length_penalty}")
    print(f"最少生成 token  : {args.min_new_tokens}")
    print(f"beam 提前停止   : {args.beam_early_stopping}")
    print(
        f"空预测          : {empty_prediction_count} "
        f"({empty_prediction_count / n * 100:.2f}%)"
    )
    print(f"图片尺寸模式    : {resize_mode}")
    if lexicon_trie:
        print(
            f"词典覆盖        : {lexicon_covered_count}/{n} "
            f"({lexicon_covered_count / n * 100:.2f}%)"
        )
    if seen_label_summary:
        for name, values in seen_label_summary.items():
            print(
                f"{name:18}: n={values['count']}, "
                f"labels={values['unique_label_count']}, "
                f"exact={values['exact_match_accuracy'] * 100:.2f}%, "
                f"CER={values['cer'] * 100:.2f}%"
            )
    if quality_summary:
        for quality, values in quality_summary.items():
            print(
                f"quality={quality:12}: n={values['count']}, "
                f"exact={values['exact_match_accuracy'] * 100:.2f}%, "
                f"CER={values['cer'] * 100:.2f}%"
            )
    print(f"重复 n-gram 约束: {args.no_repeat_ngram_size}")
    print(f"强制灰度输入    : {args.force_grayscale}")
    print("=" * 60)

    print("\n按标签长度:")
    for bucket, values in length_summary.items():
        print(
            f"  {bucket:>5}: n={values['count']:>5}, "
            f"exact={values['exact_match_accuracy'] * 100:6.2f}%, "
            f"CER={values['cer'] * 100:6.2f}%"
        )

    if set(profile_summary) != {"other"}:
        print("\n按合成数据 profile:")
        for profile, values in profile_summary.items():
            print(
                f"  {profile:>8}: n={values['count']:>5}, "
                f"exact={values['exact_match_accuracy'] * 100:6.2f}%, "
                f"CER={values['cer'] * 100:6.2f}%"
            )
    if rare_length_summary:
        print("\nrare 按标签长度:")
        for bucket, values in rare_length_summary.items():
            print(
                f"  {bucket:>5}: n={values['count']:>5}, "
                f"exact={values['exact_match_accuracy'] * 100:6.2f}%, "
                f"CER={values['cer'] * 100:6.2f}%"
            )
    if rare_metadata_summary:
        print("\nrare 按生成属性:")
        for bucket, values in rare_metadata_summary.items():
            if bucket == "_metadata_coverage":
                print(
                    "  metadata coverage: "
                    f"{values['count']}/{values['rare_count']}"
                )
                continue
            print(
                f"  {bucket:>34}: n={values['count']:>5}, "
                f"exact={values['exact_match_accuracy'] * 100:6.2f}%, "
                f"CER={values['cer'] * 100:6.2f}%"
            )

    print("\n按公司样本频次:")
    for bucket, values in frequency_summary.items():
        print(
            f"  {bucket:>6}: samples={values['sample_count']:>5}, "
            f"labels={values['unique_label_count']:>4}, "
            f"exact={values['exact_match_accuracy'] * 100:6.2f}%, "
            f"macro_exact="
            f"{values['macro_exact_match_accuracy'] * 100:6.2f}%"
        )

    if wrongs_sorted:
        print("\n错误 Top-20（按 CER 降序）:")
        for i, r in enumerate(wrongs_sorted[:20], 1):
            image_name = Path(r.image_path).name
            print(
                f"  #{i:>2} [{image_name}]  CER={r.cer * 100:5.2f}%  "
                f"label={r.label!r}  pred={r.pred!r}"
            )

    # 6. 导出报告
    report_csv = (
        Path(args.report_csv)
        if args.report_csv
        else Path("seal_dataset_eval_report.csv")
    )
    with report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exact_match",
                "cer",
                "substitutions",
                "deletions",
                "insertions",
                "label",
                "pred",
                "predicted_target_length",
                "length_confidence",
                "length_constraint_active",
                "image_path",
                "quality",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    1 if r.exact_match else 0,
                    f"{r.cer:.6f}",
                    r.substitutions,
                    r.deletions,
                    r.insertions,
                    r.label,
                    r.pred,
                    r.predicted_target_length,
                    f"{r.length_confidence:.6f}",
                    1 if r.length_constraint_active else 0,
                    r.image_path,
                    r.quality,
                ]
            )
    print(f"\n[INFO] 逐行结果已写入: {report_csv}")

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else Path("seal_dataset_eval_summary.json")
    )
    summary = {
        "model_weights": (
            args.checkpoint if args.checkpoint else args.cust_data_init_weights_path
        ),
        "dataset_source": dataset_source,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "length_condition_scale": args.length_condition_scale,
        "length_hard_constraint": not args.no_length_hard_constraint,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": args.beam_early_stopping,
        "resize_mode": resize_mode,
        "lexicon_constrained": lexicon_trie is not None,
        "lexicon_paths": args.lexicon_path,
        "lexicon_covered_count": lexicon_covered_count,
        "lexicon_coverage": lexicon_covered_count / n,
        "force_grayscale": args.force_grayscale,
        "num_samples": n,
        "exact_match_count": exact_count,
        "exact_match_accuracy": exact_acc,
        "unique_label_count": len(results_by_label),
        "macro_exact_match_accuracy": macro_exact_match,
        "zero_exact_label_count": zero_exact_label_count,
        "cer": cer,
        "macro_average_cer": macro_avg_cer,
        "unique_prediction_count": len(prediction_counts),
        "mean_label_length": mean_label_length,
        "mean_prediction_length": mean_prediction_length,
        "output_length_metrics": output_length_metrics,
        "length_head_metrics": length_head_metrics,
        "spatial_head_metrics": spatial_head_metrics,
        "length_tolerance": length_tolerance,
        "length_force_confidence": length_force_confidence,
        "empty_prediction_count": empty_prediction_count,
        "empty_prediction_rate": empty_prediction_count / n,
        "top_predictions": top_predictions,
        "error_counts": {
            "substitutions": total_substitutions,
            "deletions": total_deletions,
            "insertions": total_insertions,
        },
        "by_label_length": length_summary,
        "by_label_frequency": frequency_summary,
        "by_seen_label_status": seen_label_summary,
        "by_quality": quality_summary,
        "wrong_count": len(wrongs),
        "severe_wrong_count": severe_wrong_count,
        "very_severe_wrong_count": very_severe_wrong_count,
        "long_overrun_count": long_overrun_count,
        "wrong_top_20": [asdict(r) for r in wrongs_sorted[:20]],
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] 汇总统计已写入: {summary_json}")


if __name__ == "__main__":
    main()

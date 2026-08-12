"""分阶段训练印章公司名 TrOCR 模型。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from math import ceil, log
from pathlib import Path
from typing import Dict, List, Sequence

# Python 3.13 + PyTorch 2.9 在导入 DDP communication hooks 时触发的上游
# enum 兼容提示，与本项目训练行为无关。只过滤这一条精确消息，保留其他
# FutureWarning，避免掩盖真正需要迁移的接口。
warnings.filterwarnings(
    "ignore",
    message=(
        r"functools\.partial will be a method descriptor in future Python "
        r"versions; wrap it in enum\.member\(\) if you want to preserve the "
        r"old behavior"
    ),
    category=FutureWarning,
)
# transformers 4.51.3 的 TrOCRProcessor.from_pretrained() 内部仍通过旧参数名
# feature_extractor 恢复 image processor。项目已经使用 image_processor/processor
# 新接口，这条上游兼容提示无法由调用方消除，因此只精确过滤该消息。
warnings.filterwarnings(
    "ignore",
    message=(
        r"`feature_extractor` is deprecated and will be removed in v5\. "
        r"Use `image_processor` instead\."
    ),
    category=FutureWarning,
    module=r"transformers\.models\.trocr\.processing_trocr",
)
# VisionEncoderDecoderModel.from_pretrained() 会把共享 encoder config 覆盖
# 子模块默认配置并输出一条 info 日志，以及可能的 UserWarning。image_size 等
# 覆盖是本项目预期行为（prepare.py 已写好配置），精确屏蔽该提示。
warnings.filterwarnings(
    "ignore",
    message=(
        r"Config of the encoder: .* is overwritten by shared encoder config:"
    ),
    category=UserWarning,
    module=r"transformers",
)
# 本方案训练与选模均使用 greedy（num_beams=1），而 length_penalty 仅在
# beam 搜索时生效；保留默认值以兼容后续 beam 扫描，精确屏蔽这条兼容提示。
warnings.filterwarnings(
    "ignore",
    message=(
        r"`num_beams` is set to 1\. However, `length_penalty` is set to"
    ),
    category=UserWarning,
    module=r"transformers\.generation\.configuration_utils",
)
# TrOCRProcessor.from_pretrained() 会输出 use_fast 慢 tokenizer 切换提示；
# 本项目使用自定义 BPE 词表（prepare.py 生成），v4.52 前行为保持一致，
# 精确屏蔽这条上游迁移提示。
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
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.models.vision_encoder_decoder").setLevel(
    logging.WARNING
)
logging.getLogger("transformers.processing_utils").setLevel(logging.WARNING)

from seal_ocr.data import (
    TrainingSample,
    deduplicate_samples_by_image_content,
    discover_samples,
    fingerprint_samples,
    find_cross_split_content_duplicates,
    find_missing_characters,
    limit_samples_per_label,
    load_vocabulary,
    split_samples_by_label,
    split_samples_by_path,
    summarize_issues,
    write_split_manifest,
)


# transformers 4.51.3 中该映射名对应“logits 与 labels 同位置”的 token CE。
# 本项目的 decoder_input_ids 已由 collator 右移，不能再使用 ForCausalLM。
SEQ2SEQ_LOSS_TYPE = "ForMaskedLM"

LENGTH_BUCKETS = (
    ("len_le_12", 0, 12),
    ("len_13_16", 13, 16),
    ("len_17_19", 17, 19),
    ("len_ge_20", 20, None),
)


def set_model_output_field(outputs, name: str, value) -> None:
    """把自定义前向结果写入 ``ModelOutput`` 的真实映射键。

    Accelerate 的混合精度包装会按 ``outputs.items()`` 重建模型返回值。仅用
    ``setattr`` 增加的新属性不在映射中，BF16/FP16 前向返回后会静默丢失。
    """
    if not isinstance(name, str) or not name:
        raise ValueError("ModelOutput 字段名不能为空")
    try:
        outputs[name] = value
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError(
            f"模型返回值不支持保存自定义字段 {name!r}"
        ) from error


def get_model_output_field(outputs, model, name: str):
    """读取自定义前向字段，并兼容评估包装器的旁路缓存。

    训练前向会把字段写入 ``ModelOutput`` 映射，以便 Accelerate 和 DDP
    正确追踪梯度。某些 Transformers/Accelerate 版本在 ``generate`` 的评估
    路径会再构造一次标准输出，可能把扩展字段丢掉；前向同时保留一个按 batch
    重置的模型属性作为兜底，避免评估在有标注样本上误报“没有空间损失”。
    """
    value = getattr(outputs, name, None)
    if value is not None:
        return value
    return getattr(model, f"_last_{name}", None)


def build_auxiliary_seq2seq_output_type(base_output_type):
    """创建同时兼容 Accelerate 重建和 DDP pytree 重建的输出类型。

    BF16/FP16 包装只保留 Mapping 键，而 PyTorch 2.9 DDP 会用这些键作为
    关键字参数重新构造原输出类型。因此辅助字段既必须进入 Mapping，也必须是
    dataclass 构造函数声明过的字段。
    """

    @dataclass
    class AuxiliarySeq2SeqLMOutput(base_output_type):
        length_logits: object = None
        length_pred_loss: object = None
        spatial_loss: object = None

    return AuxiliarySeq2SeqLMOutput


def convert_to_auxiliary_model_output(outputs, output_type):
    """把标准 Seq2Seq 输出无损转换为声明了辅助字段的输出。"""
    if isinstance(outputs, output_type):
        return outputs
    if not hasattr(outputs, "items"):
        raise TypeError(
            "长度/空间辅助训练要求模型 forward 返回 Mapping 类型的 ModelOutput"
        )
    try:
        return output_type(**dict(outputs.items()))
    except TypeError as error:
        raise TypeError("无法把基础模型返回值转换为辅助 ModelOutput") from error


def compute_length_aware_metrics(predictions, references, cer_function):
    """按真实标签长度报告指标，防止短名称均值掩盖长名称归零。"""
    if len(predictions) != len(references):
        raise ValueError("predictions 与 references 数量不一致")
    metrics = {}
    bucket_exact_values = []
    long_predictions = []
    long_references = []
    for bucket_name, minimum, maximum in LENGTH_BUCKETS:
        selected = [
            (predicted, expected)
            for predicted, expected in zip(predictions, references)
            if len(expected) >= minimum
            and (maximum is None or len(expected) <= maximum)
        ]
        if not selected:
            continue
        bucket_predictions = [item[0] for item in selected]
        bucket_references = [item[1] for item in selected]
        bucket_exact = sum(
            predicted == expected
            for predicted, expected in selected
        ) / len(selected)
        metrics[f"exact_match_{bucket_name}"] = bucket_exact
        metrics[f"cer_{bucket_name}"] = cer_function(
            predictions=bucket_predictions,
            references=bucket_references,
        )
        metrics[f"samples_{bucket_name}"] = len(selected)
        bucket_exact_values.append(bucket_exact)
        if minimum >= 17:
            long_predictions.extend(bucket_predictions)
            long_references.extend(bucket_references)

    metrics["length_balanced_exact_match"] = (
        sum(bucket_exact_values) / len(bucket_exact_values)
        if bucket_exact_values
        else 0.0
    )
    if long_references:
        metrics["long_exact_match"] = sum(
            predicted == expected
            for predicted, expected in zip(long_predictions, long_references)
        ) / len(long_references)
        metrics["long_cer"] = cer_function(
            predictions=long_predictions,
            references=long_references,
        )
        metrics["long_samples"] = len(long_references)
    return metrics


def apply_decoder_context_dropout(
    torch,
    decoder_input_ids,
    probability: float,
    replacement_token_id: int,
    protected_token_ids,
):
    """随机遮住 teacher-forcing 历史字符，迫使 decoder 持续读取图像。"""
    if probability <= 0 or not decoder_input_ids.numel():
        return decoder_input_ids
    candidates = torch.ones_like(decoder_input_ids, dtype=torch.bool)
    candidates[:, 0] = False
    for token_id in protected_token_ids:
        if token_id is not None:
            candidates &= decoder_input_ids.ne(int(token_id))
    sampled = torch.rand(
        decoder_input_ids.shape,
        device=decoder_input_ids.device,
    ) < probability
    return decoder_input_ids.masked_fill(
        candidates & sampled,
        int(replacement_token_id),
    )


STAGE_DEFAULTS = {
    "pretrain": {
        "dataset_path": ["data/synthetic"],
        "num_train_epochs": 6,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "encoder_learning_rate": 2e-4,
        "cross_attention_learning_rate": 2e-4,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 25,
        "save_total_limit": 3,
        "warmup_ratio": 0.05,
        "label_smoothing_factor": 0.005,
        "early_stopping_patience": 5,
        "metric_for_best_model": "length_balanced_exact_match",
        "augmentation": "pretrain",
        "color_consistency_weight": 0.06,
        "decoder_context_dropout": 0.12,
        "degradation_enabled": True,
        "degradation_probability": 0.60,
        "text_dissolution_probability": 0.75,
        "background_clutter_probability": 0.70,
        "foreground_stroke_probability": 0.30,
        "max_text_dissolution_ratio": 0.70,
        "min_text_residual_ratio": 0.25,
        "max_foreground_text_overlap_ratio": 0.06,
        "eval_ratio": 0.02,
        "test_ratio": 0.0,
        "verify_images": False,
        "dataloader_num_workers": 12,
        "split_strategy": "label",
        "tf32": True,
        "replay_ratio": 0.0,
        "generation_num_beams": 1,
        "generation_length_penalty": 1.0,
        "generation_no_repeat_ngram_size": 0,
        "generation_min_new_tokens": 0,
        "generation_early_stopping": False,
        "deduplicate_images": False,
        "balanced_samples_per_label": 0,
        "resize_mode": "letterbox",
        "length_control_enabled": True,
        "length_pred_weight": 1.0,
        "length_predictor_hidden_size": 256,
        "length_predictor_dropout": 0.1,
        "length_tolerance": 0,
        "length_activate_steps": 1500,
        "length_learning_rate": 2e-4,
        "length_encoder_gradient_scale": 0.10,
        "length_force_confidence": 0.90,
        "spatial_control_enabled": True,
        "spatial_loss_weight": 0.20,
        "spatial_text_weight": 1.0,
        "spatial_stamp_weight": 0.5,
        "spatial_heatmap_weight": 0.25,
        "spatial_character_weight": 1.0,
        "spatial_head_hidden_size": 128,
        "spatial_learning_rate": 2e-4,
        "spatial_encoder_gradient_scale": 0.10,
        "freeze_spatial_head": False,
    },
    "finetune": {
        "dataset_path": ["data/real/train"],
        "num_train_epochs": 25,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-6,
        "encoder_learning_rate": 5e-6,
        "cross_attention_learning_rate": 5e-6,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 10,
        "save_total_limit": 3,
        "warmup_ratio": 0.10,
        "label_smoothing_factor": 0.0,
        "early_stopping_patience": 4,
        "metric_for_best_model": "macro_exact_match",
        "augmentation": "finetune",
        "color_consistency_weight": 0.03,
        "decoder_context_dropout": 0.02,
        "degradation_enabled": False,
        "degradation_probability": 0.60,
        "text_dissolution_probability": 0.75,
        "background_clutter_probability": 0.70,
        "foreground_stroke_probability": 0.30,
        "max_text_dissolution_ratio": 0.70,
        "min_text_residual_ratio": 0.25,
        "max_foreground_text_overlap_ratio": 0.06,
        "eval_ratio": 0.10,
        "test_ratio": 0.10,
        "verify_images": True,
        "dataloader_num_workers": 4,
        "split_strategy": "label",
        "tf32": True,
        "replay_ratio": 0.0,
        "generation_num_beams": 1,
        "generation_length_penalty": 1.0,
        "generation_no_repeat_ngram_size": 0,
        "generation_min_new_tokens": 0,
        "generation_early_stopping": False,
        "deduplicate_images": True,
        "balanced_samples_per_label": 0,
        "resize_mode": "letterbox",
        "length_control_enabled": True,
        "length_pred_weight": 0.3,
        "length_predictor_hidden_size": 256,
        "length_predictor_dropout": 0.1,
        "length_tolerance": 0,
        "length_activate_steps": 50,
        "length_learning_rate": 5e-6,
        "length_encoder_gradient_scale": 0.02,
        "length_force_confidence": 0.90,
        "spatial_control_enabled": True,
        "spatial_loss_weight": 0.10,
        "spatial_text_weight": 1.0,
        "spatial_stamp_weight": 0.5,
        "spatial_heatmap_weight": 0.25,
        "spatial_character_weight": 1.0,
        "spatial_head_hidden_size": 128,
        "spatial_learning_rate": 5e-6,
        "spatial_encoder_gradient_scale": 0.03,
        "freeze_spatial_head": True,
    },
}

FIXED_DEFAULTS = {
    "eval_dataset_path": None,
    "test_dataset_path": None,
    "replay_dataset_path": None,
    "spatial_annotation_path": None,
    "eval_spatial_annotation_path": None,
    "test_spatial_annotation_path": None,
    "replay_spatial_annotation_path": None,
    "allow_label_overlap": False,
    "max_samples_per_label": 0,
    "max_target_length": 40,
    "seed": 10086,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "invalid_sample_policy": "error",
    "precision": "auto",
    "gradient_checkpointing": False,
    "freeze_encoder": False,
    "resume_from_checkpoint": None,
    "overwrite_output_dir": False,
    "validate_only": False,
    "CUDA_VISIBLE_DEVICES": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="轻量印章 OCR：合成预训练或真实数据后训练"
    )
    parser.add_argument(
        "--stage",
        choices=("pretrain", "finetune"),
        default="pretrain",
        help="pretrain=合成数据预训练，finetune=真实标注后训练",
    )
    parser.add_argument(
        "--model",
        dest="cust_data_init_weights_path",
        default="models/init",
        help="初始化模型；后训练时传预训练 best 目录",
    )
    parser.add_argument(
        "--data",
        dest="dataset_path",
        nargs="+",
        default=None,
        help="训练数据目录；图片与同名 .txt 标签成对存放",
    )
    parser.add_argument(
        "--val-data",
        dest="eval_dataset_path",
        nargs="+",
        default=None,
        help="可选独立验证目录；默认按公司名稳定切分",
    )
    parser.add_argument(
        "--test-data",
        dest="test_dataset_path",
        nargs="+",
        default=None,
        help="可选独立测试目录",
    )
    parser.add_argument(
        "--replay-data",
        dest="replay_dataset_path",
        nargs="+",
        default=None,
        help="可选合成回放目录，仅用于真实后训练",
    )
    parser.add_argument(
        "--spatial-data",
        dest="spatial_annotation_path",
        nargs="+",
        default=None,
        help="合成空间标注目录；默认自动探测同级 <data>_spatial",
    )
    parser.add_argument(
        "--output",
        dest="checkpoint_path",
        default=None,
        help="训练输出目录；默认 checkpoints/<stage>",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", dest="num_train_epochs", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        dest="per_device_train_batch_size",
        type=int,
        default=None,
        help="每张 GPU 的训练 batch",
    )
    parser.add_argument(
        "--gradient-accumulation",
        dest="gradient_accumulation_steps",
        type=int,
        default=None,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--workers", dest="dataloader_num_workers", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--replay-ratio", type=float, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=None,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=None,
        help="显存不足时启用",
    )
    parser.add_argument(
        "--resume",
        dest="resume_from_checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help="从 checkpoint 续训；不传路径时自动选择最新 checkpoint",
    )
    parser.add_argument(
        "--allow-label-overlap",
        action="store_true",
        default=None,
        help="仅用于明确允许独立测试集出现训练公司名的场景",
    )
    parser.add_argument(
        "--skip-invalid",
        dest="invalid_sample_policy",
        action="store_const",
        const="skip",
        default=None,
        help="跳过损坏或缺标签样本；默认遇到问题即停止",
    )
    parser.add_argument("--verify-images", action="store_true", default=None)
    parser.add_argument("--overwrite", dest="overwrite_output_dir", action="store_true", default=None)
    parser.add_argument("--validate-only", action="store_true", default=None)
    return parser


def apply_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for defaults in (FIXED_DEFAULTS, STAGE_DEFAULTS[args.stage]):
        for key, value in defaults.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    for key in (
        "encoder_learning_rate",
        "cross_attention_learning_rate",
        "length_learning_rate",
        "spatial_learning_rate",
    ):
        setattr(args, key, args.learning_rate)
    if args.checkpoint_path is None:
        args.checkpoint_path = f"checkpoints/{args.stage}"
    return args


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def print_main(message: str) -> None:
    if is_main_process():
        print(message)


def load_and_check_samples(
    dataset_paths: Sequence[str],
    verify_images: bool,
    invalid_sample_policy: str,
    dataset_name: str,
    spatial_annotation_paths: Sequence[str] | None = None,
    auto_spatial_annotations: bool = False,
    require_spatial_annotations: bool = False,
) -> List[TrainingSample]:
    samples, issues = discover_samples(
        dataset_paths,
        verify_images=verify_images,
        spatial_annotation_paths=spatial_annotation_paths,
        auto_spatial_annotations=auto_spatial_annotations,
        require_spatial_annotations=require_spatial_annotations,
    )
    if issues:
        issue_summary = summarize_issues(issues)
        message = f"{dataset_name}发现 {len(issues)} 个数据问题:\n{issue_summary}"
        if invalid_sample_policy == "error":
            raise ValueError(message)
        print_main(f"警告：{message}\n已跳过这些样本。")
    if not samples:
        raise ValueError(f"{dataset_name}没有可用样本")
    return samples


def ensure_disjoint_labels(
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    test_samples: Sequence[TrainingSample],
) -> None:
    label_sets = {
        "train": {sample.label for sample in train_samples},
        "eval": {sample.label for sample in eval_samples},
        "test": {sample.label for sample in test_samples},
    }
    overlaps = []
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        common = label_sets[left] & label_sets[right]
        if common:
            overlaps.append(f"{left}/{right}: {sorted(common)[:10]}")
    if overlaps:
        raise ValueError(
            "不同集合存在相同公司名，评估会泄漏:\n" + "\n".join(overlaps)
        )


def ensure_disjoint_images(
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    test_samples: Sequence[TrainingSample],
) -> None:
    path_sets = {
        "train": {sample.image_path for sample in train_samples},
        "eval": {sample.image_path for sample in eval_samples},
        "test": {sample.image_path for sample in test_samples},
    }
    overlaps = []
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        common = path_sets[left] & path_sets[right]
        if common:
            overlaps.append(f"{left}/{right}: {sorted(common)[:10]}")
    if overlaps:
        raise ValueError("不同集合存在相同图片:\n" + "\n".join(overlaps))


def ensure_disjoint_image_content(
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    test_samples: Sequence[TrainingSample],
) -> None:
    """拒绝路径不同但文件内容完全相同的跨集合图片泄漏。"""
    split_samples = {
        "train": train_samples,
        "eval": eval_samples,
        "test": test_samples,
    }
    messages = []
    for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
        if not split_samples[left] or not split_samples[right]:
            continue
        overlaps = find_cross_split_content_duplicates(
            split_samples[left],
            split_samples[right],
        )
        if overlaps:
            examples = [
                f"{left}={first.image_path} <-> {right}={second.image_path}"
                for first, second in overlaps[:10]
            ]
            messages.append(
                f"{left}/{right}: {len(overlaps)} 张；" + "; ".join(examples)
            )
    if messages:
        raise ValueError(
            "不同集合存在路径不同但内容完全相同的图片，评估会泄漏:\n"
            + "\n".join(messages)
        )


def ensure_replay_images_are_new(
    replay_samples: Sequence[TrainingSample],
    other_samples: Sequence[TrainingSample],
) -> None:
    replay_paths = {sample.image_path for sample in replay_samples}
    overlapping_paths = replay_paths & {
        sample.image_path for sample in other_samples
    }
    if overlapping_paths:
        raise ValueError(
            "合成回放池与真实训练/验证/测试集存在相同图片:\n"
            + "\n".join(sorted(overlapping_paths)[:10])
        )


def summarize_split(name: str, samples: Sequence[TrainingSample]) -> str:
    labels = Counter(sample.label for sample in samples)
    return (
        f"{name}: {len(samples)} 张，{len(labels)} 个公司名，"
        f"单公司最多 {max(labels.values(), default=0)} 张"
    )


def resolve_precision(torch, precision: str) -> Dict[str, bool]:
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("当前设备不支持 bf16")
        return {"bf16": True, "fp16": False}
    if precision == "fp16":
        if not torch.cuda.is_available():
            raise ValueError("fp16 训练需要 CUDA")
        return {"bf16": False, "fp16": True}
    if precision == "fp32":
        return {"bf16": False, "fp16": False}

    if torch.cuda.is_available():
        bf16_supported = (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        return {"bf16": bf16_supported, "fp16": not bf16_supported}
    return {"bf16": False, "fp16": False}


def build_optimizer(torch, model, args):
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        target = (
            no_decay_parameters
            if name.endswith(".bias")
            or "layernorm" in lowered
            or "layer_norm" in lowered
            else decay_parameters
        )
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def load_auxiliary_checkpoint_state(torch, init_dir: Path):
    """读取自定义辅助模块权重；兼容单文件和分片 checkpoint。"""
    state = {}
    single_safetensors = init_dir / "model.safetensors"
    single_bin = init_dir / "pytorch_model.bin"
    shard_index_safetensors = init_dir / "model.safetensors.index.json"
    shard_index_bin = init_dir / "pytorch_model.bin.index.json"

    if single_safetensors.exists():
        from safetensors.torch import load_file

        state = load_file(str(single_safetensors))
    elif single_bin.exists():
        state = torch.load(
            str(single_bin), map_location="cpu", weights_only=True,
        )
    elif shard_index_safetensors.exists():
        from safetensors.torch import load_file

        index = json.loads(shard_index_safetensors.read_text("utf-8"))
        for shard_file in sorted(set(index.get("weight_map", {}).values())):
            shard_path = init_dir / shard_file
            if shard_path.exists():
                state.update(load_file(str(shard_path)))
    elif shard_index_bin.exists():
        index = json.loads(shard_index_bin.read_text("utf-8"))
        for shard_file in sorted(set(index.get("weight_map", {}).values())):
            shard_path = init_dir / shard_file
            if shard_path.exists():
                state.update(
                    torch.load(
                        str(shard_path),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
    return state


def _rebuild_tokenizer_from_vocab(
    processor, custom_vocab: dict, output_dir: str
) -> None:
    """直接修改 tokenizer.json 的 model.vocab，使 tokenizer 与 model decoder 词表对齐。

    tokenizers 库的 add_tokens 只追加到 added_tokens，不影响 base BPE vocab，
    导致 vocab_size=0。正确做法是直接覆写 tokenizer.json 中的 model.vocab 字段。
    """
    output_path = Path(output_dir)
    tokenizer_json_path = output_path / "tokenizer.json"

    # 读取现有 tokenizer.json，保留 normalizer / pre_tokenizer / post_processor / decoder
    tok_data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))

    # 替换 BPE model 的 vocab 和 merges
    tok_data["model"] = {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": custom_vocab,
        "merges": [],
    }

    # 更新 added_tokens 为自定义特殊 token（id 与 custom_vocab 一致）
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok_data["added_tokens"] = [
        {
            "id": custom_vocab[t],
            "content": t,
            "single_word": False,
            "lstrip": (t == "<mask>"),
            "rstrip": False,
            "normalized": True,
            "special": True,
        }
        for t in special_tokens
        if t in custom_vocab
    ]

    tokenizer_json_path.write_text(
        json.dumps(tok_data, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    # 重新加载 processor 并注入新 tokenizer
    from transformers import TrOCRProcessor

    new_processor = TrOCRProcessor.from_pretrained(str(output_path))
    processor.tokenizer = new_processor.tokenizer

    if processor.tokenizer.vocab_size != len(custom_vocab):
        raise RuntimeError(
            f"重建后 tokenizer.vocab_size={processor.tokenizer.vocab_size} "
            f"!= len(custom_vocab)={len(custom_vocab)}"
        )

    # 验证 vocab 映射
    probe_vocab = processor.tokenizer.get_vocab()
    for token, expected_id in custom_vocab.items():
        if probe_vocab.get(token) != expected_id:
            raise RuntimeError(
                f"重建失败：token {token} id 不匹配（期望 {expected_id}，"
                f"实际 {probe_vocab.get(token)}）"
            )

    pad_id = custom_vocab["<pad>"]
    bos_id = custom_vocab["<s>"]
    eos_id = custom_vocab["</s>"]
    print_main(
        f"已用 vocab.json 重建 tokenizer：vocab_size={processor.tokenizer.vocab_size}，"
        f"pad_id={pad_id}，bos_id={bos_id}，eos_id={eos_id}"
    )


def main() -> int:
    args = apply_stage_defaults(build_parser().parse_args())
    if args.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    if args.generation_num_beams <= 0:
        raise ValueError("generation_num_beams 必须大于 0")
    if args.generation_min_new_tokens < 0:
        raise ValueError("generation_min_new_tokens 必须大于等于 0")
    if args.generation_min_new_tokens >= args.max_target_length - 1:
        raise ValueError(
            "generation_min_new_tokens 必须小于 max_target_length - 1"
        )
    if args.balanced_samples_per_label < 0:
        raise ValueError("balanced_samples_per_label 必须大于等于 0")
    if args.stage != "finetune" and args.balanced_samples_per_label > 0:
        raise ValueError("公司名均衡过采样只用于 finetune 阶段")
    learning_rates = {
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "cross_attention_learning_rate": args.cross_attention_learning_rate,
    }
    for name, value in learning_rates.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.generation_no_repeat_ngram_size < 0:
        raise ValueError("generation_no_repeat_ngram_size 必须大于等于 0")
    if not 0 <= args.label_smoothing_factor < 1:
        raise ValueError("label_smoothing_factor 必须大于等于 0 且小于 1")
    if not 0 <= args.color_consistency_weight <= 1:
        raise ValueError("color_consistency_weight 必须在 0 到 1 之间")
    if not 0 <= args.decoder_context_dropout <= 0.5:
        raise ValueError("decoder_context_dropout 必须在 0 到 0.5 之间")
    degradation_probability_fields = {
        "degradation_probability": args.degradation_probability,
        "text_dissolution_probability": (
            args.text_dissolution_probability
        ),
        "background_clutter_probability": (
            args.background_clutter_probability
        ),
        "foreground_stroke_probability": (
            args.foreground_stroke_probability
        ),
    }
    for name, value in degradation_probability_fields.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 必须在 [0, 1] 范围内")
    if not 0 <= args.max_text_dissolution_ratio <= 0.85:
        raise ValueError(
            "max_text_dissolution_ratio 必须在 [0, 0.85] 范围内"
        )
    if not 0.20 <= args.min_text_residual_ratio <= 1.0:
        raise ValueError(
            "min_text_residual_ratio 必须在 [0.20, 1.0] 范围内"
        )
    if not 0 <= args.max_foreground_text_overlap_ratio <= 0.12:
        raise ValueError(
            "max_foreground_text_overlap_ratio 必须在 [0, 0.12] 范围内"
        )
    if args.degradation_enabled and args.augmentation != "pretrain":
        raise ValueError(
            "定向退化只在合成预训练中启用"
        )
    if args.length_control_enabled:
        if args.length_pred_weight < 0:
            raise ValueError("length_pred_weight 必须大于等于 0")
        if args.length_predictor_hidden_size <= 0:
            raise ValueError("length_predictor_hidden_size 必须大于 0")
        if not 0 <= args.length_predictor_dropout < 1:
            raise ValueError("length_predictor_dropout 必须在 [0, 1) 范围内")
        if args.length_tolerance < 0:
            raise ValueError("length_tolerance 必须大于等于 0")
        if not float(args.length_tolerance).is_integer():
            raise ValueError("length_tolerance 必须为整数")
        if args.length_activate_steps < 0:
            raise ValueError("length_activate_steps 必须大于等于 0")
        if args.length_learning_rate <= 0:
            raise ValueError("length_learning_rate 必须大于 0")
        if not 0 <= args.length_encoder_gradient_scale <= 1:
            raise ValueError(
                "length_encoder_gradient_scale 必须在 [0, 1] 范围内"
            )
        if not 0 <= args.length_force_confidence <= 1:
            raise ValueError("length_force_confidence 必须在 [0, 1] 范围内")
    if args.spatial_control_enabled:
        if args.spatial_loss_weight < 0:
            raise ValueError("spatial_loss_weight 必须大于等于 0")
        if args.spatial_text_weight <= 0 or args.spatial_stamp_weight <= 0:
            raise ValueError(
                "spatial_text_weight 和 spatial_stamp_weight 必须大于 0"
            )
        if args.spatial_heatmap_weight < 0:
            raise ValueError("spatial_heatmap_weight 必须大于等于 0")
        if args.spatial_character_weight <= 0:
            raise ValueError("spatial_character_weight 必须大于 0")
        if args.spatial_head_hidden_size <= 0:
            raise ValueError("spatial_head_hidden_size 必须大于 0")
        if args.spatial_learning_rate <= 0:
            raise ValueError("spatial_learning_rate 必须大于 0")
        if not 0 <= args.spatial_encoder_gradient_scale <= 1:
            raise ValueError(
                "spatial_encoder_gradient_scale 必须在 [0, 1] 范围内"
            )
    elif not args.degradation_enabled and any(
        paths is not None
        for paths in (
            args.spatial_annotation_path,
            args.eval_spatial_annotation_path,
            args.test_spatial_annotation_path,
            args.replay_spatial_annotation_path,
        )
    ):
        raise ValueError(
            "关闭空间控制和预训练退化时不能指定空间标注目录"
        )
    if args.eval_spatial_annotation_path and not args.eval_dataset_path:
        raise ValueError(
            "eval_spatial_annotation_path 只能与 eval_dataset_path 一起使用"
        )
    if args.test_spatial_annotation_path and not args.test_dataset_path:
        raise ValueError(
            "test_spatial_annotation_path 只能与 test_dataset_path 一起使用"
        )

    use_primary_spatial_annotations = (
        args.spatial_control_enabled or args.degradation_enabled
    )
    require_train_spatial = (
        args.degradation_enabled
        or (
            args.spatial_control_enabled
            and args.stage == "pretrain"
        )
    )
    require_eval_spatial = (
        args.spatial_control_enabled and args.stage == "pretrain"
    )

    all_samples = load_and_check_samples(
        args.dataset_path,
        verify_images=args.verify_images,
        invalid_sample_policy=args.invalid_sample_policy,
        dataset_name="训练数据",
        spatial_annotation_paths=args.spatial_annotation_path,
        auto_spatial_annotations=use_primary_spatial_annotations,
        require_spatial_annotations=require_train_spatial,
    )

    def maybe_deduplicate(
        samples: Sequence[TrainingSample],
        dataset_name: str,
    ) -> List[TrainingSample]:
        if not args.deduplicate_images:
            return list(samples)
        unique_samples, duplicates = deduplicate_samples_by_image_content(samples)
        if duplicates:
            print_main(
                f"{dataset_name}按图片内容去重: {len(samples)} -> "
                f"{len(unique_samples)}，排除 {len(duplicates)} 个重复路径"
            )
        return unique_samples

    all_samples = maybe_deduplicate(all_samples, "训练数据")
    independent_test_samples = (
        maybe_deduplicate(
            load_and_check_samples(
                args.test_dataset_path,
                verify_images=args.verify_images,
                invalid_sample_policy=args.invalid_sample_policy,
                dataset_name="测试数据",
                spatial_annotation_paths=args.test_spatial_annotation_path,
                auto_spatial_annotations=args.spatial_control_enabled,
                require_spatial_annotations=require_eval_spatial,
            ),
            "测试数据",
        )
        if args.test_dataset_path
        else []
    )

    if args.eval_dataset_path:
        train_samples = all_samples
        eval_samples = maybe_deduplicate(
            load_and_check_samples(
                args.eval_dataset_path,
                verify_images=args.verify_images,
                invalid_sample_policy=args.invalid_sample_policy,
                dataset_name="验证数据",
                spatial_annotation_paths=args.eval_spatial_annotation_path,
                auto_spatial_annotations=args.spatial_control_enabled,
                require_spatial_annotations=require_eval_spatial,
            ),
            "验证数据",
        )
        test_samples = independent_test_samples
    else:
        if independent_test_samples and args.test_ratio > 0:
            raise ValueError(
                "已指定独立 test_dataset_path 时 test_ratio 必须为 0"
            )
        split_function = (
            split_samples_by_label
            if args.split_strategy == "label"
            else split_samples_by_path
        )
        train_samples, eval_samples, internal_test_samples = split_function(
            all_samples,
            eval_ratio=args.eval_ratio,
            test_ratio=0 if independent_test_samples else args.test_ratio,
            seed=args.seed,
        )
        test_samples = independent_test_samples or internal_test_samples

    ensure_disjoint_images(train_samples, eval_samples, test_samples)
    # 对真实数据量级执行内容哈希审计；十万级合成预训练未指定独立集合时不额外
    # 扫描整池，避免启动训练前重复读取所有 JPEG。
    if (
        args.deduplicate_images
        or args.eval_dataset_path
        or args.test_dataset_path
    ):
        ensure_disjoint_image_content(train_samples, eval_samples, test_samples)
    if not args.allow_label_overlap:
        ensure_disjoint_labels(train_samples, eval_samples, test_samples)
    elif not args.eval_dataset_path:
        # 内部分出的 eval 始终保持公司名隔离；只允许独立 test 与训练出现同名公司。
        ensure_disjoint_labels(train_samples, eval_samples, [])

    train_samples = limit_samples_per_label(
        train_samples,
        max_samples_per_label=args.max_samples_per_label,
        seed=args.seed,
    )

    if not 0 <= args.replay_ratio < 1:
        raise ValueError("replay_ratio 必须大于等于 0 且小于 1")
    if args.replay_dataset_path and args.replay_ratio <= 0:
        raise ValueError("指定 replay_dataset_path 时 replay_ratio 必须大于 0")
    if args.replay_ratio > 0 and not args.replay_dataset_path:
        raise ValueError("replay_ratio 大于 0 时必须指定 replay_dataset_path")
    if args.stage != "finetune" and args.replay_dataset_path:
        raise ValueError("合成回放只用于 finetune 阶段")
    if args.replay_spatial_annotation_path and not args.replay_dataset_path:
        raise ValueError(
            "指定 replay_spatial_annotation_path 时必须同时指定 replay_dataset_path"
        )
    if args.stage == "finetune" and args.spatial_control_enabled:
        if args.replay_dataset_path and args.replay_ratio > 0.50:
            print_main(
                "警告：合成回放超过 50%，真实域梯度会被明显稀释；80% 只适合"
                "空间保持能力诊断，不建议作为最终真实后训练配置。"
            )
        elif args.replay_dataset_path and not 0.15 <= args.replay_ratio <= 0.35:
            print_main(
                "警告：空间保持回放通常建议 15%–35%，首选 20%–30%；"
                f"当前为 {args.replay_ratio:.1%}。"
            )

    replay_samples: List[TrainingSample] = []
    if args.replay_dataset_path:
        replay_samples = load_and_check_samples(
            args.replay_dataset_path,
            verify_images=False,
            invalid_sample_policy=args.invalid_sample_policy,
            dataset_name="合成回放数据",
            spatial_annotation_paths=args.replay_spatial_annotation_path,
            auto_spatial_annotations=args.spatial_control_enabled,
            require_spatial_annotations=args.spatial_control_enabled,
        )
        ensure_replay_images_are_new(
            replay_samples,
            [*train_samples, *eval_samples, *test_samples],
        )

    vocab_path = Path(args.cust_data_init_weights_path) / "vocab.json"
    vocabulary = load_vocabulary(str(vocab_path))
    all_checked_samples = [
        *train_samples,
        *eval_samples,
        *test_samples,
        *replay_samples,
    ]
    if args.spatial_control_enabled:
        from seal_ocr.spatial_annotations import spatial_detail_annotation_path

        missing_detail_annotations = [
            str(spatial_detail_annotation_path(sample.spatial_annotation_path))
            for sample in all_checked_samples
            if sample.spatial_annotation_path is not None
            and not spatial_detail_annotation_path(
                sample.spatial_annotation_path
            ).is_file()
        ]
        if missing_detail_annotations:
            raise FileNotFoundError(
                "空间监督需要每张主标注对应 .detail.png；"
                f"当前缺少 {len(missing_detail_annotations)} 张，示例: "
                f"{missing_detail_annotations[:5]}。请用当前生成器重做空间标注。"
            )
    missing_characters = find_missing_characters(
        all_checked_samples,
        vocabulary,
    )
    if missing_characters:
        raise ValueError(
            "模型词表未覆盖以下标签字符；请先更新词表并重新初始化模型: "
            f"{missing_characters}"
        )
    maximum_label_length = max(
        len(sample.label)
        for sample in all_checked_samples
    )
    if maximum_label_length > args.max_target_length - 2:
        examples = [
            sample.label
            for sample in all_checked_samples
            if len(sample.label) == maximum_label_length
        ][:5]
        raise ValueError(
            f"最长标签为 {maximum_label_length} 字，但 max_target_length="
            f"{args.max_target_length} 只能容纳 {args.max_target_length - 2} 字。"
            f"示例: {examples}"
        )

    print_main(summarize_split("训练集", train_samples))
    print_main(summarize_split("验证集", eval_samples))
    print_main(summarize_split("测试集", test_samples))
    if args.spatial_control_enabled or args.degradation_enabled:
        for split_name, split_samples in (
            ("训练集", train_samples),
            ("验证集", eval_samples),
            ("测试集", test_samples),
        ):
            annotated_count = sum(
                sample.spatial_annotation_path is not None
                for sample in split_samples
            )
            print_main(
                f"{split_name}空间标注: {annotated_count}/{len(split_samples)} "
                f"({annotated_count / max(1, len(split_samples)):.1%})"
            )
    if args.degradation_enabled:
        print_main("预训练定向退化已启用；验证和测试保持原图。")
    eval_length_buckets = {
        bucket_name: sum(
            len(sample.label) >= minimum
            and (maximum is None or len(sample.label) <= maximum)
            for sample in eval_samples
        )
        for bucket_name, minimum, maximum in LENGTH_BUCKETS
    }
    print_main(f"验证集长度分桶: {eval_length_buckets}")
    if args.metric_for_best_model == "length_balanced_exact_match":
        empty_buckets = [
            name for name, count in eval_length_buckets.items() if count <= 0
        ]
        if empty_buckets:
            raise ValueError(
                f"{args.metric_for_best_model} 要求验证集覆盖全部长度桶；"
                f"当前缺少 {empty_buckets}。请提高 eval_ratio 或补充相应长度数据。"
            )
    if replay_samples:
        primary_per_epoch = (
            len({sample.label for sample in train_samples})
            * args.balanced_samples_per_label
            if args.balanced_samples_per_label > 0
            else len(train_samples)
        )
        mixed_total = ceil(primary_per_epoch / (1 - args.replay_ratio))
        replay_per_epoch = mixed_total - primary_per_epoch
        print_main(summarize_split("合成回放池", replay_samples))
        if args.spatial_control_enabled:
            annotated_replay_count = sum(
                sample.spatial_annotation_path is not None
                for sample in replay_samples
            )
            print_main(
                "合成回放空间标注: "
                f"{annotated_replay_count}/{len(replay_samples)}"
            )
        print_main(
            f"每个 epoch: 真实增强 {primary_per_epoch} 张 + "
            f"轮换合成约 {replay_per_epoch:.0f} 张，"
            f"回放比例 {args.replay_ratio:.1%}"
        )

    if args.save_steps % args.eval_steps != 0:
        raise ValueError(
            "load_best_model_at_end 要求 save_steps 是 eval_steps 的整数倍"
        )

    if args.validate_only:
        print_main(
            f"数据、词表与 {args.split_strategy} 切分检查通过；"
            "未加载模型，未启动训练。"
        )
        return 0

    output_dir = Path(args.checkpoint_path)
    manifest_path = output_dir / "data_split.json"
    manifest_settings = {
        "dataset_path": args.dataset_path,
        "eval_dataset_path": args.eval_dataset_path,
        "test_dataset_path": args.test_dataset_path,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "test_ratio": args.test_ratio,
        "split_strategy": args.split_strategy,
        "max_samples_per_label": args.max_samples_per_label,
        "balanced_samples_per_label": args.balanced_samples_per_label,
        "deduplicate_images": args.deduplicate_images,
        "resize_mode": args.resize_mode,
        "allow_label_overlap": args.allow_label_overlap,
        "replay_dataset_path": args.replay_dataset_path,
        "replay_ratio": args.replay_ratio,
        "augmentation": args.augmentation,
        "color_consistency_weight": args.color_consistency_weight,
        "decoder_context_dropout": args.decoder_context_dropout,
        "metric_for_best_model": args.metric_for_best_model,
        "length_control_enabled": args.length_control_enabled,
        "length_pred_weight": args.length_pred_weight,
        "length_predictor_hidden_size": args.length_predictor_hidden_size,
        "length_predictor_dropout": args.length_predictor_dropout,
        "length_tolerance": args.length_tolerance,
        "length_activate_steps": args.length_activate_steps,
        "length_learning_rate": args.length_learning_rate,
        "length_encoder_gradient_scale": args.length_encoder_gradient_scale,
        "length_force_confidence": args.length_force_confidence,
    }
    if args.augmentation == "pretrain":
        manifest_settings.update(
            {
                "degradation_enabled": args.degradation_enabled,
                "degradation_probability": args.degradation_probability,
                "text_dissolution_probability": (
                    args.text_dissolution_probability
                ),
                "background_clutter_probability": (
                    args.background_clutter_probability
                ),
                "foreground_stroke_probability": (
                    args.foreground_stroke_probability
                ),
                "max_text_dissolution_ratio": (
                    args.max_text_dissolution_ratio
                ),
                "min_text_residual_ratio": (
                    args.min_text_residual_ratio
                ),
                "max_foreground_text_overlap_ratio": (
                    args.max_foreground_text_overlap_ratio
                ),
            }
        )
        if args.degradation_enabled:
            manifest_settings["spatial_annotation_path"] = (
                args.spatial_annotation_path
            )
    if args.spatial_control_enabled:
        manifest_settings.update(
            {
                "spatial_control_enabled": True,
                "spatial_annotation_path": args.spatial_annotation_path,
                "eval_spatial_annotation_path": (
                    args.eval_spatial_annotation_path
                ),
                "test_spatial_annotation_path": (
                    args.test_spatial_annotation_path
                ),
                "replay_spatial_annotation_path": (
                    args.replay_spatial_annotation_path
                ),
                "spatial_loss_weight": args.spatial_loss_weight,
                "spatial_text_weight": args.spatial_text_weight,
                "spatial_stamp_weight": args.spatial_stamp_weight,
                "spatial_heatmap_weight": args.spatial_heatmap_weight,
                "spatial_character_weight": args.spatial_character_weight,
                "spatial_head_hidden_size": args.spatial_head_hidden_size,
                "spatial_learning_rate": args.spatial_learning_rate,
                "spatial_encoder_gradient_scale": (
                    args.spatial_encoder_gradient_scale
                ),
                "freeze_spatial_head": args.freeze_spatial_head,
            }
        )
    current_fingerprints = {
        "train": fingerprint_samples(train_samples),
        "eval": fingerprint_samples(eval_samples),
        "test": fingerprint_samples(test_samples),
    }
    if replay_samples:
        current_fingerprints["replay"] = fingerprint_samples(replay_samples)
    # 只有 rank 0 检查并写入 manifest，避免多进程竞争
    if is_main_process():
        if manifest_path.exists():
            if args.resume_from_checkpoint is None:
                if not args.overwrite_output_dir:
                    raise FileExistsError(
                        f"checkpoint 目录已有数据切分: {manifest_path}。"
                        "请使用新目录，或明确传入 --resume_from_checkpoint auto，"
                        "或启用 --overwrite_output_dir 覆盖。"
                    )
            else:
                previous_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if (
                    previous_manifest.get("settings") != manifest_settings
                    or previous_manifest.get("fingerprints") != current_fingerprints
                ):
                    raise ValueError(
                        "续训时的数据、标签或切分参数与原 checkpoint 不一致；"
                        "为避免指标失真，已停止续训。"
                    )
        # 覆盖模式且非续训：清空已有 checkpoint 目录后重新开始
        if (
            args.overwrite_output_dir
            and args.resume_from_checkpoint is None
            and output_dir.exists()
        ):
            print_main(
                f"checkpoint 目录已存在，启用覆盖模式，清理旧目录: {output_dir}"
            )
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_split_manifest(
            str(manifest_path),
            train_samples,
            eval_samples,
            test_samples,
            settings=manifest_settings,
            extra_splits={"replay": replay_samples} if replay_samples else None,
        )

    # CUDA 可见设备已设置完毕，此后再加载 torch/transformers。
    import numpy as np
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from transformers.modeling_outputs import Seq2SeqLMOutput

    AuxiliarySeq2SeqLMOutput = build_auxiliary_seq2seq_output_type(
        Seq2SeqLMOutput
    )

    from seal_ocr.dataset import (
        LabelBalancedDataset,
        ReplayMixDataset,
        decode_text,
        trocrDataset,
    )
    from seal_ocr.metrics import compute_cer
    from seal_ocr.augment import get_augmentation
    from seal_ocr.image import processor_image_size
    from seal_ocr.degradation import DegradationConfig
    from seal_ocr.length_control import (
        LengthConditionModule,
        LengthControlLogitsProcessor,
        condition_activation_ratio,
        compute_length_prediction_loss,
        compute_true_lengths,
        infer_current_step,
        scale_gradient,
    )
    from seal_ocr.spatial_control import (
        SpatialAuxiliaryHead,
        compute_spatial_objective,
        infer_patch_grid,
    )
    from seal_ocr.spatial_annotations import SPATIAL_CHANNELS

    class TrOCRDataCollator:
        """TrOCR 专用 collator：把 labels 移位为 decoder_input_ids，
        并让 VisionEncoderDecoderModel 在 transformers 4.51+ 中正确接收。

        注意：vocab_size 必须使用 model.decoder.config.vocab_size，
        而不是 tokenizer.vocab_size，否则自定义词表模型会出现 CUDA 越界。"""

        def __init__(
            self,
            pad_token_id: int,
            decoder_start_token_id: int,
            vocab_size: int,
        ):
            self.pad_token_id = pad_token_id
            self.decoder_start_token_id = decoder_start_token_id
            self.vocab_size = vocab_size

            if not 0 <= pad_token_id < vocab_size:
                raise ValueError(
                    f"pad_token_id={pad_token_id} 超出词表范围 [0, {vocab_size})"
                )
            if not 0 <= decoder_start_token_id < vocab_size:
                raise ValueError(
                    "decoder_start_token_id="
                    f"{decoder_start_token_id} 超出词表范围 [0, {vocab_size})"
                )

        def __call__(self, features):
            pixel_values = torch.stack([f["pixel_values"] for f in features])
            labels = torch.stack([f["labels"] for f in features])
            invariant_views = [
                feature.get("color_invariant_pixel_values")
                for feature in features
            ]
            has_invariant_views = [view is not None for view in invariant_views]
            if any(has_invariant_views) and not all(has_invariant_views):
                raise ValueError("同一 batch 的颜色一致性视图不完整")
            spatial_targets = [
                feature.get("spatial_targets") for feature in features
            ]
            spatial_flags = [
                feature.get("has_spatial_annotation") for feature in features
            ]
            has_spatial_fields = [
                target is not None and flag is not None
                for target, flag in zip(spatial_targets, spatial_flags)
            ]
            # 外部 eval 数据集可能只有部分样本带空间标注，甚至完全没有
            # 标注。只要 batch 中至少有一条完整字段，就用同形状的全零 target
            # 和 False 门控补齐缺失项；这样有标注样本仍会训练空间头，真实样本
            # 不会因为没有区域信息而让整个 eval batch 失败。完全没有字段时
            # 保持旧行为，不把空间监督伪造为全零负样本。
            if any(has_spatial_fields):
                template_target = next(
                    target
                    for target, complete in zip(
                        spatial_targets,
                        has_spatial_fields,
                    )
                    if complete
                )
                normalized_targets = []
                normalized_flags = []
                for target, flag, complete in zip(
                    spatial_targets,
                    spatial_flags,
                    has_spatial_fields,
                ):
                    if not complete:
                        normalized_targets.append(torch.zeros_like(template_target))
                        normalized_flags.append(
                            torch.zeros((), dtype=torch.bool)
                        )
                    else:
                        if target.shape != template_target.shape:
                            raise ValueError(
                                "同一 batch 的空间 target 形状不一致: "
                                f"expected={tuple(template_target.shape)}, "
                                f"actual={tuple(target.shape)}"
                            )
                        normalized_targets.append(target)
                        normalized_flags.append(
                            torch.as_tensor(flag, dtype=torch.bool).reshape(())
                        )
                spatial_targets = normalized_targets
                spatial_flags = normalized_flags

            invalid_labels = (labels < 0) & (labels != -100)
            invalid_labels |= labels >= self.vocab_size
            if invalid_labels.any():
                invalid_values = torch.unique(labels[invalid_labels]).tolist()
                raise ValueError(
                    "labels 含词表范围外的 token id，拒绝静默 clamp: "
                    f"{invalid_values[:20]}，vocab_size={self.vocab_size}"
                )

            # 与 VisionEncoderDecoderModel.shift_tokens_right 完全一致：
            # decoder_input_ids 首位必须是 decoder_start_token_id，而不是 PAD。
            shifted = labels.new_full(labels.shape, self.pad_token_id)
            shifted[:, 1:] = labels[:, :-1]
            shifted[:, 0] = self.decoder_start_token_id
            # 把 -100（ignore_index）替换为 pad_token_id，避免 embedding 查表越界
            decoder_input_ids = shifted.masked_fill(
                shifted == -100,
                self.pad_token_id,
            )
            batch = {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
            }
            if all(has_invariant_views):
                batch["color_invariant_pixel_values"] = torch.stack(
                    invariant_views
                )
            if any(has_spatial_fields):
                batch["spatial_targets"] = torch.stack(spatial_targets)
                batch["has_spatial_annotation"] = torch.stack(spatial_flags)
            return batch

    # accelerate 的多卡验证会把最后一个 global batch 补齐到相同尺寸。
    # transformers 4.51.3 的 evaluation_loop 把实际 num_samples 用于吞吐统计，
    # 但 compute_metrics 仍可能收到补齐后的 predictions/labels。记录当前评估集
    # 的真实长度，在计算 CER/exact_match 前显式裁掉末尾补样。
    metric_sample_limit = {"value": len(eval_samples)}
    reported_metric_padding = set()

    class SealSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(
            self,
            *trainer_args,
            color_consistency_weight=0.0,
            decoder_context_dropout=0.0,
            decoder_context_replacement_token_id=None,
            decoder_context_protected_token_ids=(),
            length_control_enabled=False,
            length_pred_weight=0.0,
            length_activate_steps=0,
            length_tolerance=0,
            length_encoder_gradient_scale=0.0,
            length_force_confidence=0.90,
            length_ignored_token_ids=(),
            eos_token_id=None,
            bos_token_id=None,
            spatial_control_enabled=False,
            spatial_loss_weight=0.0,
            **trainer_kwargs,
        ):
            self.color_consistency_weight = float(color_consistency_weight)
            self.decoder_context_dropout = float(decoder_context_dropout)
            self.decoder_context_replacement_token_id = (
                int(decoder_context_replacement_token_id)
                if decoder_context_replacement_token_id is not None
                else None
            )
            self.decoder_context_protected_token_ids = tuple(
                token_id
                for token_id in decoder_context_protected_token_ids
                if token_id is not None
            )
            self.length_control_enabled = bool(length_control_enabled)
            self.length_pred_weight = float(length_pred_weight)
            # 预测长度条件在 activate_steps 内从 0 平滑增强到 1。
            self.length_activate_steps = int(length_activate_steps)
            self.length_tolerance = int(length_tolerance)
            self.length_encoder_gradient_scale = float(
                length_encoder_gradient_scale
            )
            self.length_force_confidence = float(length_force_confidence)
            self.length_ignored_token_ids = tuple(
                int(token_id)
                for token_id in length_ignored_token_ids
                if token_id is not None
            )
            self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None
            self.bos_token_id = int(bos_token_id) if bos_token_id is not None else None
            self.spatial_control_enabled = bool(spatial_control_enabled)
            self.spatial_loss_weight = float(spatial_loss_weight)
            if (
                self.decoder_context_dropout > 0
                and self.decoder_context_replacement_token_id is None
            ):
                raise ValueError("decoder context dropout 缺少替换 token id")
            if self.length_control_enabled and self.eos_token_id is None:
                raise ValueError("启用长度控制时必须提供 eos_token_id")
            if self.spatial_control_enabled and self.spatial_loss_weight < 0:
                raise ValueError("spatial_loss_weight 必须大于等于 0")
            super().__init__(*trainer_args, **trainer_kwargs)
            # VisionEncoderDecoderModel.forward 有 **kwargs，Trainer 会因此认为
            # model 自己处理 gradient accumulation 的总 token 数。本类的复合
            # 损失是按每个 micro-batch 求均值，应让 Trainer 统一除以累积步数。
            self.model_accepts_loss_kwargs = False
        def log(self, logs, start_time=None):
            visible_fields = {
                "loss",
                "grad_norm",
                "learning_rate",
                "epoch",
                "train_loss",
                "eval_loss",
                "eval_cer",
                "eval_exact_match",
                "eval_macro_exact_match",
            }
            public_logs = {
                key: value
                for key, value in logs.items()
                if key in visible_fields
            }
            return super().log(public_logs, start_time=start_time)

        def _apply_length_control(self, model, outputs, labels):
            """计算长度条件模块的损失。

            返回 (supervised_loss, length_loss)。

            长度信息已在 forward 中加入 decoder hidden 并重新投影，本方法
            只负责计算监督损失和长度预测损失：

            - supervised_loss：token CE（包含 EOS，不移位，ForMaskedLM 对齐）
            - length_loss：长度预测 CE 损失 × weight
            - 总损失 = supervised_loss + length_loss

            重要：length_pred_loss 在 forward 内部计算（attached to outputs），
            确保 length_predictor 梯度完全来自 forward 输出的 backward 图，
            避免 DDP "marked ready twice" 错误。
            """
            unwrapped = getattr(model, "module", model)

            logits = outputs.logits
            # length_logits 已在 forward 内部通过 length_control 计算
            length_logits = get_model_output_field(
                outputs,
                unwrapped,
                "length_logits",
            )
            if labels is not None and length_logits is None:
                raise RuntimeError(
                    "启用长度控制时前向缺少 length_logits；"
                    "自定义 ModelOutput 字段可能在混合精度包装中丢失"
                )

            # 监督损失：token CE；logits 已同时包含 hidden 条件和 EOS 条件。
            if labels is not None:
                supervised_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=self.label_smoother.epsilon
                    if self.label_smoother is not None
                    else 0.0,
                )
            else:
                supervised_loss = torch.tensor(
                    0.0, device=logits.device, requires_grad=True,
                )

            # 长度预测 CE 损失：从 forward 输出获取（已在 forward 内计算）
            is_training = bool(model.training) and labels is not None
            length_loss = torch.tensor(
                0.0, device=logits.device, requires_grad=False,
            )
            if is_training and length_logits is not None:
                length_pred_loss = get_model_output_field(
                    outputs,
                    unwrapped,
                    "length_pred_loss",
                )
                if length_pred_loss is None:
                    raise RuntimeError(
                        "训练前向缺少 length_pred_loss，长度头将无法监督"
                    )
                length_loss = self.length_pred_weight * length_pred_loss
            return supervised_loss, length_loss

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            # 这些是数据管线专用字段，不能直接传给
            # VisionEncoderDecoderModel.forward()。
            invariant_pixel_values = inputs.get("color_invariant_pixel_values")
            spatial_targets = inputs.get("spatial_targets")
            has_spatial_annotation = inputs.get("has_spatial_annotation")
            model_inputs = {
                key: value
                for key, value in inputs.items()
                if key
                not in {
                    "color_invariant_pixel_values",
                    "spatial_targets",
                    "has_spatial_annotation",
                }
            }
            if (
                self.spatial_control_enabled
                and bool(model.training)
                and (
                    spatial_targets is None
                    or has_spatial_annotation is None
                )
            ):
                raise RuntimeError("启用空间控制时 batch 缺少空间监督门控字段")
            if model.training and self.decoder_context_dropout > 0:
                decoder_input_ids = model_inputs.get("decoder_input_ids")
                if decoder_input_ids is None:
                    raise RuntimeError(
                        "启用 decoder_context_dropout 时 batch 缺少 "
                        "decoder_input_ids"
                    )
                model_inputs["decoder_input_ids"] = (
                    apply_decoder_context_dropout(
                        torch,
                        decoder_input_ids,
                        probability=self.decoder_context_dropout,
                        replacement_token_id=(
                            self.decoder_context_replacement_token_id
                        ),
                        protected_token_ids=(
                            self.decoder_context_protected_token_ids
                        ),
                    )
                )

            use_color_consistency = (
                invariant_pixel_values is not None
                and self.color_consistency_weight > 0
                and model.training
            )
            use_length_control = self.length_control_enabled
            use_spatial_control = (
                self.spatial_control_enabled
                and spatial_targets is not None
                and has_spatial_annotation is not None
            )

            # 无任何特殊训练功能时走原生路径
            if (
                not use_color_consistency
                and not use_length_control
                and not use_spatial_control
            ):
                return super().compute_loss(
                    model,
                    model_inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            batch_size = model_inputs["pixel_values"].shape[0]
            if use_color_consistency:
                combined_inputs = {}
                for key, value in model_inputs.items():
                    if key == "pixel_values":
                        combined_inputs[key] = torch.cat(
                            [value, invariant_pixel_values], dim=0
                        )
                    elif (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == batch_size
                    ):
                        combined_inputs[key] = torch.cat([value, value], dim=0)
                    else:
                        combined_inputs[key] = value
            else:
                combined_inputs = model_inputs

            labels = combined_inputs.get("labels")
            phase = "train" if bool(model.training) else "eval"
            unwrapped = getattr(model, "module", model)
            if use_spatial_control:
                unwrapped._spatial_supervision_context = (
                    spatial_targets,
                    has_spatial_annotation,
                    batch_size,
                )

            # ---- 前向 ----
            try:
                if use_length_control:
                    # 在剥离 labels 前计算长度监督和预测条件激活强度，供 forward
                    # 包装使用。真实长度只用于独立分类损失，不进入 decoder。
                    if labels is not None:
                        true_lengths = compute_true_lengths(
                            labels, self.bos_token_id, self.eos_token_id,
                        )
                        global_step = int(self.state.global_step)
                        current_condition_scale = condition_activation_ratio(
                            global_step,
                            self.length_activate_steps,
                        )
                        unwrapped._length_condition_context = (
                            true_lengths,
                            current_condition_scale,
                        )

                    # 不传 labels：避免 model 内部用未偏置 logits 算 loss
                    forward_inputs = {
                        key: value
                        for key, value in combined_inputs.items()
                        if key != "labels"
                    }
                    outputs = model(**forward_inputs)
                    supervised_loss, length_loss = self._apply_length_control(
                        model, outputs, labels,
                    )
                    total_loss = supervised_loss + length_loss
                else:
                    if self.label_smoother is not None and labels is not None:
                        forward_inputs = {
                            key: value
                            for key, value in combined_inputs.items()
                            if key != "labels"
                        }
                        outputs = model(**forward_inputs)
                        supervised_loss = self.label_smoother(outputs, labels)
                    else:
                        outputs = model(**combined_inputs)
                        supervised_loss = outputs.loss
                    total_loss = supervised_loss
            finally:
                # 外部监督只在本次前向有效，绝不能泄漏到 generate/下一 batch。
                for context_name in (
                    "_length_condition_context",
                    "_spatial_supervision_context",
                ):
                    if hasattr(unwrapped, context_name):
                        delattr(unwrapped, context_name)

            # ---- 空间辅助损失 ----
            if use_spatial_control and bool(has_spatial_annotation.any()):
                spatial_loss = get_model_output_field(
                    outputs,
                    unwrapped,
                    "spatial_loss",
                )
                if spatial_loss is None:
                    raise RuntimeError(
                        "有空间标注的 batch 未返回空间损失，空间头将无法训练"
                    )
                total_loss = total_loss + (
                    self.spatial_loss_weight * spatial_loss
                )

            # ---- 颜色一致性损失 ----
            if use_color_consistency:
                primary_logits = outputs.logits[:batch_size]
                invariant_logits = outputs.logits[batch_size:]
                primary_log_probs = torch.nn.functional.log_softmax(
                    primary_logits.float(), dim=-1
                )
                invariant_log_probs = torch.nn.functional.log_softmax(
                    invariant_logits.float(), dim=-1
                )
                primary_probs = primary_log_probs.exp()
                invariant_probs = invariant_log_probs.exp()
                primary_kl = torch.nn.functional.kl_div(
                    primary_log_probs,
                    invariant_probs.detach(),
                    reduction="none",
                ).sum(dim=-1)
                invariant_kl = torch.nn.functional.kl_div(
                    invariant_log_probs,
                    primary_probs.detach(),
                    reduction="none",
                ).sum(dim=-1)
                valid_tokens = model_inputs["labels"].ne(-100)
                consistency_loss = (
                    0.5 * (primary_kl + invariant_kl) * valid_tokens
                ).sum().clamp_min(0) / valid_tokens.sum().clamp_min(1)

                encoder_features = getattr(
                    outputs, "encoder_last_hidden_state", None,
                )
                if encoder_features is None:
                    raise RuntimeError(
                        "当前模型未返回 encoder_last_hidden_state，"
                        "无法执行颜色不变性训练"
                    )
                primary_features = torch.nn.functional.normalize(
                    encoder_features[:batch_size].float(), dim=-1,
                )
                invariant_features = torch.nn.functional.normalize(
                    encoder_features[batch_size:].float(), dim=-1,
                )
                encoder_consistency_loss = (
                    1.0 - (primary_features * invariant_features).sum(dim=-1)
                ).clamp_min(0).mean()
                total_loss = total_loss + (
                    self.color_consistency_weight
                    * (consistency_loss + 0.25 * encoder_consistency_loss)
                )

            if return_outputs:
                # 恢复为原 batch 大小，避免下游把双视图当成双倍样本
                if use_color_consistency:
                    outputs.logits = outputs.logits[:batch_size]
                outputs.loss = total_loss
                return total_loss, outputs
            return total_loss

        def _patch_generate_for_length_control(self, model):
            """临时包装 model.generate，施加高置信度精确长度约束。

            forward 已应用可学习的 hidden/EOS 软条件；这里仅在长度头达到
            置信度阈值时阻止提前 EOS，并在目标时间步精确结束。激活阶段把
            置信度乘以同一个 condition scale，避免软条件还很弱时硬约束先介入。

            返回还原函数。仅在 predict_with_generate 的验证/测试路径生效。
            """
            # 无论是否启用长度头，都要剥离只供训练使用的空间/颜色字段，避免
            # Seq2SeqTrainer 把它们透传给 generate 后触发未知参数错误。
            unwrapped = getattr(model, "module", model)
            original_generate = model.generate
            eos_token_id = self.eos_token_id
            tolerance = self.length_tolerance
            minimum_confidence = self.length_force_confidence
            length_control = (
                unwrapped.length_control
                if self.length_control_enabled
                else None
            )

            def patched_generate(*args, **kwargs):
                # 剥离 generate 不支持的 kwargs
                kwargs.pop("labels", None)
                kwargs.pop("color_invariant_pixel_values", None)
                kwargs.pop("spatial_targets", None)
                kwargs.pop("has_spatial_annotation", None)

                if length_control is None:
                    return original_generate(*args, **kwargs)

                # 获取 pixel_values：优先 kwargs，其次 args[0]
                pixel_values = kwargs.get("pixel_values")
                if pixel_values is None and args:
                    pixel_values = args[0]

                condition_scale = condition_activation_ratio(
                    int(self.state.global_step),
                    self.length_activate_steps,
                )

                # 重新计算 predicted_length：确保与 generate 的 batch_size
                # 一致（不能使用 forward 包装缓存的 _last_predicted_length，
                # 因为训练时 forward 可能是颜色一致性双视图 batch_size=64，
                # 而 generate 时是单视图 batch_size=32，维度不匹配）。
                # 在 eval 模式下调用 encoder 是安全的：DDP 在 eval 模式下
                # 不做梯度同步，不会导致 NCCL 状态不一致。
                if pixel_values is not None:
                    with torch.no_grad():
                        encoder_outputs = unwrapped.encoder(
                            pixel_values=pixel_values,
                        )
                        encoder_hidden = encoder_outputs.last_hidden_state
                        length_logits = length_control(
                            encoder_hidden.detach()
                        )
                        predicted_length, confidence = length_control.decode(
                            length_logits
                        )
                    effective_confidence = confidence * condition_scale
                    processor = LengthControlLogitsProcessor(
                        predicted_length=predicted_length,
                        confidence=effective_confidence,
                        eos_token_id=eos_token_id,
                        tolerance=tolerance,
                        minimum_confidence=minimum_confidence,
                        ignored_token_ids=self.length_ignored_token_ids,
                    )
                    existing = kwargs.get("logits_processor")
                    if existing is None:
                        kwargs["logits_processor"] = [processor]
                    else:
                        kwargs["logits_processor"] = [
                            *existing, processor,
                        ]

                missing = object()
                previous_scale = getattr(
                    unwrapped,
                    "_length_condition_scale",
                    missing,
                )
                unwrapped._length_condition_scale = condition_scale
                try:
                    return original_generate(*args, **kwargs)
                finally:
                    if previous_scale is missing:
                        delattr(unwrapped, "_length_condition_scale")
                    else:
                        unwrapped._length_condition_scale = previous_scale

            model.generate = patched_generate

            def restore():
                if "generate" in model.__dict__:
                    del model.__dict__["generate"]

            return restore

        def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None,
        ):
            restore = self._patch_generate_for_length_control(model)
            try:
                # 这些字段只服务于训练辅助损失，不能进入生成式 eval。
                auxiliary_ignore_keys = {
                    "length_logits",
                    "length_pred_loss",
                    "spatial_loss",
                }
                effective_ignore_keys = list(
                    dict.fromkeys(
                        [
                            *(ignore_keys or ()),
                            *auxiliary_ignore_keys,
                        ]
                    )
                )
                step_result = super().prediction_step(
                    model, inputs, prediction_loss_only,
                    ignore_keys=effective_ignore_keys,
                )
                return step_result
            finally:
                restore()

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            active_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )
            if hasattr(active_dataset, "__len__"):
                metric_sample_limit["value"] = len(active_dataset)
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        def predict(
            self,
            test_dataset,
            ignore_keys=None,
            metric_key_prefix="test",
        ):
            if hasattr(test_dataset, "__len__"):
                metric_sample_limit["value"] = len(test_dataset)
            return super().predict(
                test_dataset=test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

    set_seed(args.seed)
    tf32_enabled = False
    if args.tf32 and torch.cuda.is_available():
        capability_major, _ = torch.cuda.get_device_capability()
        tf32_enabled = capability_major >= 8
        if tf32_enabled:
            # PyTorch 2.9 起旧 allow_tf32 属性会触发弃用告警；兼容旧版的同时
            # 优先使用新精度控制 API。TrainingArguments 传 None，避免其再次
            # 调用旧接口覆盖这里的设置。
            if (
                hasattr(torch.backends.cuda.matmul, "fp32_precision")
                and hasattr(torch.backends.cudnn.conv, "fp32_precision")
            ):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            print_main("当前 CUDA 设备不支持 TF32，已自动关闭。")
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)

    # -------- 关键：保证 tokenizer 与 model decoder 使用同一份自定义词表 --------
    # 先加载 model，用 model.decoder.vocab_size 作为权威值
    model = VisionEncoderDecoderModel.from_pretrained(
        args.cust_data_init_weights_path
    )
    model_decoder_vocab_size = model.config.decoder.vocab_size

    custom_vocab_path = Path(args.cust_data_init_weights_path) / "vocab.json"
    custom_vocab = None
    if custom_vocab_path.exists():
        custom_vocab = json.loads(
            custom_vocab_path.read_text(encoding="utf-8")
        )

    tokenizer_vocab_size = processor.tokenizer.vocab_size
    tokenizer_vocab = processor.tokenizer.get_vocab()
    if custom_vocab is not None and len(custom_vocab) == model_decoder_vocab_size:
        tokenizer_mapping_matches = all(
            tokenizer_vocab.get(token) == token_id
            for token, token_id in custom_vocab.items()
        )
        if (
            tokenizer_vocab_size != model_decoder_vocab_size
            or not tokenizer_mapping_matches
        ):
            print_main(
                f"检测到 tokenizer.vocab_size={tokenizer_vocab_size} 与 "
                f"model.decoder.vocab_size={model_decoder_vocab_size} "
                "或 token-id 映射不一致，"
                "用 vocab.json 重建 tokenizer。"
            )
            # 只允许主 rank 写 tokenizer 文件；同步后所有 rank 重新加载。
            if is_main_process():
                _rebuild_tokenizer_from_vocab(
                    processor, custom_vocab, args.cust_data_init_weights_path
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            processor = TrOCRProcessor.from_pretrained(
                args.cust_data_init_weights_path
            )
        else:
            print_main(
                f"tokenizer / model 词表一致 (vocab_size={tokenizer_vocab_size})"
            )
    else:
        print_main(
            f"未找到可用 vocab.json（或 size 不匹配），"
            f"沿用 tokenizer 现有词表 (vocab_size={tokenizer_vocab_size})。"
        )

    # 同步 config 中特殊 token id，确保生成时使用正确
    special_token_ids = {
        "bos_token_id": processor.tokenizer.cls_token_id,
        "decoder_start_token_id": processor.tokenizer.cls_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.sep_token_id,
    }
    for key, token_id in special_token_ids.items():
        setattr(model.config, key, token_id)
        setattr(model.config.decoder, key, token_id)
        setattr(model.decoder.config, key, token_id)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder.use_cache = not args.gradient_checkpointing
    model.config.seal_resize_mode = args.resize_mode

    # 数据集 labels 已显式包含 BOS/EOS，collator 又把 labels 右移后作为
    # decoder_input_ids，因此 decoder logits 与原始 labels 是逐位置对齐的。
    # transformers 4.51.3 默认回退到 ForCausalLMLoss，会把 labels 再左移一次；
    # 让后训练与启用 label smoothing 的预训练使用同一套 token 对齐方式。
    # 使用两套不同目标，表现为初始化模型生成正确、loss 却接近 12，随后输出
    # 长度减半并归零。ForMaskedLMLoss 在该版本中就是所需的未移位 token CE。
    # 若调用方在这里之前访问过 loss_function，Transformers 会缓存 setter 结果；
    # 清掉该可选缓存，保证旧 checkpoint 的配置不会残留到当前训练。
    if hasattr(model, "_loss_function"):
        delattr(model, "_loss_function")
    model.loss_type = SEQ2SEQ_LOSS_TYPE
    model.config.loss_type = SEQ2SEQ_LOSS_TYPE
    resolved_loss_name = getattr(model.loss_function, "__name__", "")
    if resolved_loss_name != "ForMaskedLMLoss":
        raise RuntimeError(
            "当前 transformers 未解析到未移位 token CE："
            f"loss_type={SEQ2SEQ_LOSS_TYPE}, resolved={resolved_loss_name!r}"
        )
    print_main(
        "监督损失对齐: labels 与 decoder logits 同位置计算 "
        f"({resolved_loss_name})"
    )

    # transformers 的 generate() 优先读取 model.generation_config。只改
    # model.config 会受初始化目录中旧 generation_config.json 的影响，导致训练
    # 验证、独立评测和部署使用不同的长度惩罚/重复约束，因此两处都显式同步。
    generation_config = model.generation_config
    for key, token_id in special_token_ids.items():
        setattr(generation_config, key, token_id)
    generation_config.max_length = args.max_target_length
    generation_config.min_new_tokens = args.generation_min_new_tokens
    generation_config.num_beams = args.generation_num_beams
    generation_config.no_repeat_ngram_size = (
        args.generation_no_repeat_ngram_size
    )
    generation_config.length_penalty = args.generation_length_penalty
    generation_config.early_stopping = args.generation_early_stopping
    print_main(
        "验证生成配置: "
        + json.dumps(
            {
                "bos_token_id": generation_config.bos_token_id,
                "decoder_start_token_id": generation_config.decoder_start_token_id,
                "eos_token_id": generation_config.eos_token_id,
                "max_length": generation_config.max_length,
                "min_new_tokens": generation_config.min_new_tokens,
                "num_beams": generation_config.num_beams,
                "length_penalty": generation_config.length_penalty,
                "early_stopping": generation_config.early_stopping,
                "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    # 初始化权重可能仍把旧式生成参数写在 model.config 中。训练/部署只读取
    # generation_config；清空旧字段可避免 save_pretrained 再迁移并告警。
    for generation_parameter in (
        "max_length",
        "min_new_tokens",
        "num_beams",
        "no_repeat_ngram_size",
        "length_penalty",
        "early_stopping",
    ):
        setattr(model.config, generation_parameter, None)

    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    auxiliary_checkpoint_state = {}
    if args.length_control_enabled or args.spatial_control_enabled:
        auxiliary_checkpoint_state = load_auxiliary_checkpoint_state(
            torch,
            Path(args.cust_data_init_weights_path),
        )

    # ---- 长度条件模块 ----
    # 1. 长度头从 encoder 预测离散字符数；
    # 2. 剩余长度加入 decoder 顶层 hidden 后重新做词表投影；
    # 3. decoder 始终使用 stop-gradient 的预测长度，不接收真实长度 oracle；
    # 4. 预测条件按 smoothstep 从 0 激活到 1；
    # 5. 长度辅助梯度按较小比例回传视觉 encoder。
    if args.length_control_enabled:
        encoder_hidden_size = model.config.encoder.hidden_size
        decoder_hidden_size = model.config.decoder.hidden_size

        # 后训练从预训练模型恢复长度头；首次预训练则随机初始化。
        saved_lc_config = getattr(model.config, "length_control_config", None)
        has_saved_lc_config = bool(
            saved_lc_config and saved_lc_config.get("enabled", False)
        )
        if args.stage != "pretrain" and not has_saved_lc_config:
            raise ValueError(
                "后训练模型缺少已训练长度头；请先完成合成预训练。"
            )
        if has_saved_lc_config:
            saved_architecture_version = int(
                saved_lc_config.get("architecture_version", 0)
            )
            if saved_architecture_version != 3:
                raise ValueError(
                    "当前项目只接受 length_control architecture v3；"
                    f"实际为 v{saved_architecture_version}"
                )
            if saved_lc_config.get("conditioning_mode") != "predicted_detached":
                raise ValueError("长度头 conditioning_mode 必须是 predicted_detached")
            lc_predictor_hidden = saved_lc_config.get(
                "predictor_hidden_size", args.length_predictor_hidden_size,
            )
            lc_predictor_dropout = saved_lc_config.get(
                "predictor_dropout", args.length_predictor_dropout,
            )
            lc_max_target_length = saved_lc_config.get(
                "max_target_length", args.max_target_length,
            )
            print_main(
                "从 model.config 检测到已有长度条件配置，"
                "使用保存的架构参数创建模块"
            )
        else:
            lc_predictor_hidden = args.length_predictor_hidden_size
            lc_predictor_dropout = args.length_predictor_dropout
            lc_max_target_length = args.max_target_length

        # 创建长度条件模块
        model.length_control = LengthConditionModule(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            max_target_length=lc_max_target_length,
            predictor_hidden_size=lc_predictor_hidden,
            predictor_dropout=lc_predictor_dropout,
        )
        # 将配置写入 model.config，使 save_pretrained 后可恢复
        model.config.length_control_config = {
            "enabled": True,
            "architecture_version": 3,
            "conditioning_mode": "predicted_detached",
            "max_target_length": lc_max_target_length,
            "predictor_hidden_size": lc_predictor_hidden,
            "predictor_dropout": lc_predictor_dropout,
            "activate_steps": int(args.length_activate_steps),
            "encoder_gradient_scale": args.length_encoder_gradient_scale,
            "force_confidence": args.length_force_confidence,
            "tolerance": int(args.length_tolerance),
        }

        # 尝试从初始化目录加载已有长度条件权重
        lc_keys = {
            key.replace("length_control.", "", 1): value
            for key, value in auxiliary_checkpoint_state.items()
            if key.startswith("length_control.")
        }
        if lc_keys:
            try:
                model.length_control.load_state_dict(lc_keys, strict=True)
                print_main("已从初始化权重加载长度条件模块权重")
            except (RuntimeError, ValueError) as load_error:
                raise RuntimeError(
                    "长度头权重与当前固定 v3 结构不兼容"
                ) from load_error
        elif has_saved_lc_config:
            raise RuntimeError(
                "model.config 声明启用了长度头，但 checkpoint 缺少 length_control 权重"
            )
        else:
            print_main("长度条件模块随机初始化（初始化权重中无对应参数）")
        print_main("长度控制已启用（固定配置）。")

        # ---- 包装 forward：在 logits 输出层添加长度条件 ----
        # 流程：
        # 1. 获取或计算 encoder 输出
        # 2. 以受控梯度从 encoder 输出计算 length_logits
        # 3. 计算预测长度条件的平滑激活比例
        # 4. 调用原始 forward
        # 5. 以 detached 预测分布计算 hidden/EOS 有界残差
        # 6. 附加 length_logits 和 length_pred_loss
        #
        # DDP 兼容：长度条件通过 apply_length_condition 纳入 logits 的
        # autograd 图，DDP 能从 logits 遍历检测到 length_control 参数被使用。
        _original_forward = model.forward
        _length_module = model.length_control
        _bos_id = model.config.decoder.bos_token_id
        _eos_id = model.config.decoder.eos_token_id
        _activate_steps = args.length_activate_steps
        _encoder_gradient_scale = args.length_encoder_gradient_scale
        _output_projection = model.get_output_embeddings()
        if _output_projection is None:
            raise RuntimeError("当前 decoder 没有输出投影，无法应用长度 hidden 条件")

        def _forward_with_length_control(*args, **kwargs):
            # 评估包装器可能在 DDP/Accelerate 之后重建标准 ModelOutput；
            # 旁路缓存必须按每次 forward 清空，不能沿用上一 batch 的图。
            model._last_length_logits = None
            model._last_length_pred_loss = None
            kwargs.setdefault("output_hidden_states", True)
            kwargs["return_dict"] = True

            # 优先从外部 context 读取真实长度监督和条件激活强度
            # （compute_loss 在剥离 labels 前设置）。真实长度不会传入条件模块。
            external_context = getattr(model, "_length_condition_context", None)
            labels = kwargs.get("labels")
            if external_context is not None:
                true_lengths, current_condition_scale = external_context
            else:
                true_lengths = None
                if labels is not None:
                    true_lengths = compute_true_lengths(
                        labels, _bos_id, _eos_id,
                    )
                explicit_scale = getattr(
                    model,
                    "_length_condition_scale",
                    None,
                )
                if explicit_scale is not None:
                    current_condition_scale = float(explicit_scale)
                else:
                    trainer_ref = getattr(model, "_trainer_ref", None)
                    if trainer_ref is not None:
                        global_step = int(trainer_ref.state.global_step)
                        current_condition_scale = condition_activation_ratio(
                            global_step,
                            _activate_steps,
                        )
                    else:
                        # 独立推理默认使用完整的预测长度条件。
                        current_condition_scale = 1.0

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
                length_encoder_input = scale_gradient(
                    encoder_hidden,
                    _encoder_gradient_scale if bool(model.training) else 0.0,
                )
                length_logits = _length_module(length_encoder_input)

            # 调用原始 forward
            outputs = convert_to_auxiliary_model_output(
                _original_forward(*args, **kwargs),
                AuxiliarySeq2SeqLMOutput,
            )

            # 在 logits 输出层添加长度条件
            if length_logits is not None and outputs.logits is not None:
                # 获取最后一个 decoder layer 的 hidden states
                # hidden_states 是 tuple，最后一个元素是顶层输出
                hidden_states = None
                if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
                    hidden_states = outputs.decoder_hidden_states[-1]
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]

                if hidden_states is not None:
                    # 推断当前步数：generate use_cache=True 时 logits 的 T=1，
                    # 但需要知道实际位置才能正确计算 remaining = L - t
                    current_step = infer_current_step(kwargs)

                    # 加入 hidden 条件后重新执行 decoder 输出投影，确保全部
                    # 字符 logits 和 hidden_bias_embedding 都真正参与梯度。
                    new_logits, _ = _length_module.apply_length_condition(
                        logits=outputs.logits,
                        hidden_states=hidden_states,
                        length_logits=length_logits,
                        eos_token_id=_eos_id,
                        output_projection=_output_projection,
                        current_step=current_step,
                        condition_scale=current_condition_scale,
                    )
                    outputs.logits = new_logits

                # 附加 length_logits 供 _apply_length_control 使用
                set_model_output_field(outputs, "length_logits", length_logits)
                model._last_length_logits = length_logits

                # 记录 predicted_length 供 generate 的 LogitsProcessor 使用
                with torch.no_grad():
                    predicted_length, confidence = _length_module.decode(
                        length_logits
                    )
                    model._last_predicted_length = predicted_length
                    model._last_length_confidence = confidence

                # 在 forward 内部计算 length_pred_loss：
                # DDP find_unused_parameters=True 要求参数梯度来自 forward
                # 输出的 backward 图。如果 CE loss 在 compute_loss（forward
                # 外部）计算，DDP 会检测到 length_predictor 在 forward 图中
                # 使用但梯度来自外部，导致 grad hook 触发两次（marked ready
                # twice）。将 CE loss 移入 forward 可确保梯度路径完全在
                # forward 内部。
                if (
                    true_lengths is not None
                    and bool(model.training)
                    and external_context is not None
                ):
                    length_pred_loss = compute_length_prediction_loss(
                        length_logits,
                        true_lengths,
                        _length_module.max_target_length,
                    )
                    set_model_output_field(
                        outputs,
                        "length_pred_loss",
                        length_pred_loss,
                    )
                    model._last_length_pred_loss = length_pred_loss
                else:
                    set_model_output_field(outputs, "length_pred_loss", None)

            return outputs

        model.forward = _forward_with_length_control

    # ---- 文字/章体空间辅助模块 ----
    # 空间头只读取 encoder patch token，不参与 decoder 或 generate。真实样本的
    # has_spatial_annotation=False 时完全跳过空间头；真实后训练默认冻结头参数，
    # 但带标注合成回放的空间损失仍以小梯度穿过冻结头约束 encoder。
    spatial_grid_size = None
    spatial_channel_count = len(SPATIAL_CHANNELS)
    if args.spatial_control_enabled:
        target_width, target_height = processor_image_size(processor)
        spatial_grid_size = infer_patch_grid(
            model.config.encoder,
            (target_height, target_width),
        )
        saved_spatial_config = getattr(
            model.config,
            "spatial_control_config",
            None,
        )
        has_saved_spatial_config = bool(
            saved_spatial_config
            and saved_spatial_config.get("enabled", False)
        )
        if args.stage != "pretrain" and not has_saved_spatial_config:
            raise ValueError(
                f"{args.stage} 启用了空间控制，但初始化权重不含已训练空间头。"
                "请先完成合成预训练。"
            )
        if has_saved_spatial_config:
            saved_architecture_version = int(
                saved_spatial_config.get("architecture_version", 0)
            )
            if saved_architecture_version != 2:
                raise ValueError(
                    "当前项目只接受八通道空间头 checkpoint；"
                    f"实际 architecture_version={saved_architecture_version}"
                )
            saved_output_channels = int(
                saved_spatial_config.get("output_channels", 0)
            )
            if saved_output_channels != spatial_channel_count:
                raise ValueError(
                    "空间头 config 必须是八通道，实际为 "
                    f"{saved_output_channels}"
                )
            saved_grid_size = tuple(
                int(value)
                for value in saved_spatial_config.get(
                    "grid_size",
                    spatial_grid_size,
                )
            )
            if saved_grid_size != spatial_grid_size:
                raise ValueError(
                    "初始化空间头网格与当前 processor/encoder 不一致: "
                    f"saved={saved_grid_size}, current={spatial_grid_size}"
                )
            spatial_head_hidden_size = int(
                saved_spatial_config.get(
                    "head_hidden_size",
                    args.spatial_head_hidden_size,
                )
            )
        else:
            spatial_head_hidden_size = args.spatial_head_hidden_size

        model.spatial_head = SpatialAuxiliaryHead(
            encoder_hidden_size=model.config.encoder.hidden_size,
            head_hidden_size=spatial_head_hidden_size,
            grid_size=spatial_grid_size,
        )
        spatial_keys = {
            key.replace("spatial_head.", "", 1): value
            for key, value in auxiliary_checkpoint_state.items()
            if key.startswith("spatial_head.")
        }
        if spatial_keys:
            try:
                model.spatial_head.load_state_dict(spatial_keys, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    "空间头权重与当前固定八通道结构不兼容"
                ) from error
            print_main("已从初始化权重加载空间辅助头")
        elif has_saved_spatial_config:
            raise RuntimeError(
                "model.config 声明已启用空间头，但 checkpoint 缺少 spatial_head 权重"
            )
        elif args.freeze_spatial_head:
            raise ValueError("不能冻结随机初始化的空间头")
        else:
            print_main("空间辅助头随机初始化（首次空间预训练）")

        if args.freeze_spatial_head:
            for parameter in model.spatial_head.parameters():
                parameter.requires_grad = False

        model.config.spatial_control_config = {
            "enabled": True,
            "architecture_version": 2,
            "supervision_mode": "patch_auxiliary_character_order_v2",
            "channels": list(SPATIAL_CHANNELS),
            "output_channels": spatial_channel_count,
            "grid_size": list(spatial_grid_size),
            "head_hidden_size": spatial_head_hidden_size,
            "loss_weight": args.spatial_loss_weight,
            "text_weight": args.spatial_text_weight,
            "stamp_weight": args.spatial_stamp_weight,
            "heatmap_weight": args.spatial_heatmap_weight,
            "character_weight": args.spatial_character_weight,
            "encoder_gradient_scale": args.spatial_encoder_gradient_scale,
            "head_frozen_during_stage": bool(args.freeze_spatial_head),
        }
        print_main("空间辅助监督已启用（固定配置）。")

        _forward_before_spatial_control = model.forward
        _spatial_head = model.spatial_head
        _spatial_encoder_gradient_scale = (
            args.spatial_encoder_gradient_scale
        )

        def _forward_with_spatial_control(*forward_args, **forward_kwargs):
            # 与长度头相同，旁路值只覆盖当前 forward，避免评估批次之间
            # 读取到上一批的空间损失。
            model._last_spatial_loss = None
            spatial_context = getattr(
                model,
                "_spatial_supervision_context",
                None,
            )
            if spatial_context is None:
                return _forward_before_spatial_control(
                    *forward_args,
                    **forward_kwargs,
                )

            encoder_outputs = forward_kwargs.get("encoder_outputs")
            if encoder_outputs is None:
                pixel_values = forward_kwargs.get("pixel_values")
                if pixel_values is None and forward_args:
                    pixel_values = forward_args[0]
                if pixel_values is None:
                    raise RuntimeError("空间监督前向缺少 pixel_values")
                encoder_outputs = model.encoder(pixel_values=pixel_values)
                forward_kwargs["encoder_outputs"] = encoder_outputs

            outputs = convert_to_auxiliary_model_output(
                _forward_before_spatial_control(
                    *forward_args,
                    **forward_kwargs,
                ),
                AuxiliarySeq2SeqLMOutput,
            )
            targets, annotation_flags, primary_batch_size = spatial_context
            annotation_flags = annotation_flags.bool().reshape(-1)
            if annotation_flags.numel() != int(primary_batch_size):
                raise RuntimeError(
                    "空间标注门控 batch 大小不一致: "
                    f"flags={annotation_flags.numel()}, "
                    f"primary_batch={primary_batch_size}"
            )
            set_model_output_field(outputs, "spatial_loss", None)
            if bool(annotation_flags.any()):
                encoder_hidden = getattr(
                    encoder_outputs,
                    "last_hidden_state",
                    None,
                )
                if encoder_hidden is None:
                    encoder_hidden = encoder_outputs[0]
                primary_hidden = encoder_hidden[: int(primary_batch_size)]
                annotated_hidden = primary_hidden[annotation_flags]
                annotated_targets = targets[annotation_flags]
                spatial_encoder_input = scale_gradient(
                    annotated_hidden,
                    _spatial_encoder_gradient_scale
                    if bool(model.training)
                    else 0.0,
                )
                spatial_logits = _spatial_head(spatial_encoder_input)
                spatial_loss, _ = compute_spatial_objective(
                    spatial_logits,
                    annotated_targets,
                    text_weight=args.spatial_text_weight,
                    stamp_weight=args.spatial_stamp_weight,
                    heatmap_weight=args.spatial_heatmap_weight,
                    character_weight=args.spatial_character_weight,
                    return_metrics=False,
                )
                # 空间损失必须作为 ModelOutput 的真实映射键返回。这样 DDP 的
                # find_unused_parameters 能从标准 forward 输出遍历到空间头，
                # 同时 Accelerate 重建混合精度返回值时不会丢失该梯度路径。
                set_model_output_field(outputs, "spatial_loss", spatial_loss)
                model._last_spatial_loss = spatial_loss
            return outputs

        model.forward = _forward_with_spatial_control
    else:
        model.config.spatial_control_config = {"enabled": False}

    # checkpoint state 可能包含完整基础模型，辅助模块恢复完毕后立即释放。
    auxiliary_checkpoint_state = None

    vocab = processor.tokenizer.get_vocab()
    expected_token_ids = set(range(model_decoder_vocab_size))
    actual_token_ids = set(vocab.values())
    if actual_token_ids != expected_token_ids:
        missing_ids = sorted(expected_token_ids - actual_token_ids)[:20]
        extra_ids = sorted(actual_token_ids - expected_token_ids)[:20]
        raise ValueError(
            "tokenizer token id 与 decoder 词表不连续或不一致："
            f"missing={missing_ids}, extra={extra_ids}, "
            f"tokenizer_size={len(vocab)}, decoder_size={model_decoder_vocab_size}"
        )
    if custom_vocab is not None and vocab != custom_vocab:
        mismatched_tokens = [
            token
            for token, token_id in custom_vocab.items()
            if vocab.get(token) != token_id
        ][:20]
        raise ValueError(
            "tokenizer 与 vocab.json 的 token-id 映射不一致："
            f"{mismatched_tokens}"
        )
    vocab_inverse = {token_id: token for token, token_id in vocab.items()}

    if args.label_smoothing_factor > 0:
        epsilon = args.label_smoothing_factor
        target_probability = 1 - epsilon + epsilon / model_decoder_vocab_size
        other_probability = epsilon / model_decoder_vocab_size
        smoothing_loss_floor = -target_probability * log(target_probability)
        smoothing_loss_floor -= (
            (model_decoder_vocab_size - 1)
            * other_probability
            * log(other_probability)
        )
        print_main(
            f"label_smoothing={epsilon:g}，按当前 vocab_size="
            f"{model_decoder_vocab_size}，理论最小交叉熵约 "
            f"{smoothing_loss_floor:.4f}；请主要观察 CER/exact_match。"
        )

    def compute_metrics(prediction):
        prediction_ids = prediction.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        label_ids = np.array(prediction.label_ids, copy=True)

        expected_count = metric_sample_limit["value"]
        if len(prediction_ids) != len(label_ids):
            raise RuntimeError(
                "验证 predictions/labels 数量不一致: "
                f"{len(prediction_ids)} != {len(label_ids)}"
            )
        gathered_count = len(label_ids)
        if expected_count and gathered_count < expected_count:
            raise RuntimeError(
                "多卡验证聚合样本少于验证集: "
                f"gathered={gathered_count}, actual={expected_count}"
            )
        if expected_count and gathered_count > expected_count:
            warning_key = (gathered_count, expected_count)
            if warning_key not in reported_metric_padding:
                print_main(
                    "多卡验证末批补样已从指标中排除: "
                    f"gathered={gathered_count}, actual={expected_count}"
                )
                reported_metric_padding.add(warning_key)
            prediction_ids = prediction_ids[:expected_count]
            label_ids = label_ids[:expected_count]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        metric_values = {}
        gathered_losses = getattr(prediction, "losses", None)
        if gathered_losses is not None:
            gathered_losses = np.asarray(gathered_losses).reshape(-1)
            if expected_count and len(gathered_losses) > expected_count:
                gathered_losses = gathered_losses[:expected_count]
            metric_values["unpadded_loss"] = float(gathered_losses.mean())

        predictions = [
            decode_text(tokens, vocab, vocab_inverse)
            for tokens in prediction_ids
        ]
        references = [
            decode_text(tokens, vocab, vocab_inverse)
            for tokens in label_ids
        ]
        cer = compute_cer(predictions=predictions, references=references)
        exact_match = sum(
            predicted == expected
            for predicted, expected in zip(predictions, references)
        ) / max(len(references), 1)
        grouped_correct: Dict[str, List[bool]] = {}
        for predicted, expected in zip(predictions, references):
            grouped_correct.setdefault(expected, []).append(predicted == expected)
        macro_exact_match = sum(
            sum(values) / len(values)
            for values in grouped_correct.values()
        ) / max(len(grouped_correct), 1)
        empty_prediction_count = sum(not value for value in predictions)
        mean_prediction_length = sum(map(len, predictions)) / max(
            len(predictions), 1
        )
        length_errors = [
            len(predicted) - len(expected)
            for predicted, expected in zip(predictions, references)
        ]
        length_count = max(len(length_errors), 1)
        prediction_counts = Counter(predictions)
        top_prediction_share = (
            prediction_counts.most_common(1)[0][1] / len(predictions)
            if predictions
            else 0.0
        )
        metric_values.update(
            {
                "cer": cer,
                "exact_match": exact_match,
                "macro_exact_match": macro_exact_match,
                "empty_prediction_rate": empty_prediction_count
                / max(len(predictions), 1),
                "mean_prediction_length": mean_prediction_length,
                "output_length_accuracy": sum(
                    error == 0 for error in length_errors
                )
                / length_count,
                "output_length_mae": sum(
                    abs(error) for error in length_errors
                )
                / length_count,
                "output_length_within_1": sum(
                    abs(error) <= 1 for error in length_errors
                )
                / length_count,
                "output_length_under_rate": sum(
                    error < 0 for error in length_errors
                )
                / length_count,
                "output_length_over_rate": sum(
                    error > 0 for error in length_errors
                )
                / length_count,
                "top_prediction_share": top_prediction_share,
            }
        )
        metric_values.update(
            compute_length_aware_metrics(
                predictions,
                references,
                cer_function=compute_cer,
            )
        )
        return metric_values

    decoder_context_replacement_token_id = processor.tokenizer.mask_token_id
    if decoder_context_replacement_token_id is None:
        decoder_context_replacement_token_id = processor.tokenizer.unk_token_id
    if (
        args.decoder_context_dropout > 0
        and decoder_context_replacement_token_id is None
    ):
        raise ValueError(
            "decoder_context_dropout 需要 tokenizer 提供 <mask> 或 <unk> token"
        )
    decoder_context_protected_token_ids = tuple(
        token_id
        for token_id in (
            processor.tokenizer.bos_token_id,
            processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id,
            decoder_context_replacement_token_id,
        )
        if token_id is not None
    )
    color_consistency_enabled = args.color_consistency_weight > 0
    degradation_config = DegradationConfig(
        enabled=args.degradation_enabled,
        probability=args.degradation_probability,
        text_dissolution_probability=(
            args.text_dissolution_probability
        ),
        background_clutter_probability=(
            args.background_clutter_probability
        ),
        foreground_stroke_probability=(
            args.foreground_stroke_probability
        ),
        max_text_dissolution_ratio=(
            args.max_text_dissolution_ratio
        ),
        min_text_residual_ratio=args.min_text_residual_ratio,
        max_foreground_text_overlap_ratio=(
            args.max_foreground_text_overlap_ratio
        ),
    )
    primary_train_dataset = trocrDataset(
        paths=train_samples,
        processor=processor,
        max_target_length=args.max_target_length,
        transformer=get_augmentation(
            args.augmentation,
            degradation_config=degradation_config,
        ),
        color_invariant_view=color_consistency_enabled,
        resize_mode=args.resize_mode,
        spatial_target_size=spatial_grid_size,
        load_spatial_annotation_for_augmentation=(
            args.degradation_enabled and args.augmentation == "pretrain"
        ),
    )
    if args.balanced_samples_per_label > 0:
        primary_train_dataset = LabelBalancedDataset(
            dataset=primary_train_dataset,
            samples=train_samples,
            samples_per_label=args.balanced_samples_per_label,
            seed=args.seed,
        )
        print_main(
            "公司名均衡采样: "
            f"{len({sample.label for sample in train_samples})} 个公司 × "
            f"{args.balanced_samples_per_label} 张/epoch = "
            f"{len(primary_train_dataset)} 张"
        )
    if replay_samples:
        replay_dataset = trocrDataset(
            paths=replay_samples,
            processor=processor,
            max_target_length=args.max_target_length,
            transformer=get_augmentation("pretrain"),
            color_invariant_view=color_consistency_enabled,
            resize_mode=args.resize_mode,
            spatial_target_size=spatial_grid_size,
        )
        train_dataset = ReplayMixDataset(
            primary_dataset=primary_train_dataset,
            replay_dataset=replay_dataset,
            replay_ratio=args.replay_ratio,
            seed=args.seed,
        )
    else:
        train_dataset = primary_train_dataset
    eval_dataset = trocrDataset(
        paths=eval_samples,
        processor=processor,
        max_target_length=args.max_target_length,
        resize_mode=args.resize_mode,
        spatial_target_size=spatial_grid_size,
    )
    test_dataset = (
        trocrDataset(
            paths=test_samples,
            processor=processor,
            max_target_length=args.max_target_length,
            resize_mode=args.resize_mode,
            spatial_target_size=spatial_grid_size,
        )
        if test_samples
        else None
    )

    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    print_main(
        f"训练配置: stage={args.stage}, learning_rate={args.learning_rate:g}; "
        "参数量="
        + json.dumps(parameter_counts, ensure_ascii=False, separators=(",", ":"))
    )

    precision = resolve_precision(torch, args.precision)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.generation_num_beams,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.metric_for_best_model != "cer",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing_factor,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        tf32=None,
        seed=args.seed,
        data_seed=args.seed,
        include_for_metrics=["loss"],
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=True,
    )

    optimizer = build_optimizer(torch, model, args)
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.00001,
        )
    ]
    if hasattr(train_dataset, "set_epoch"):
        class EpochAwareDatasetCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                train_dataset.set_epoch(int(state.epoch or 0))

        callbacks.append(EpochAwareDatasetCallback())

    trainer = None
    baseline_metric_value = None
    try:
        trainer = SealSeq2SeqTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=TrOCRDataCollator(
                pad_token_id=processor.tokenizer.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id,
                vocab_size=model.config.decoder.vocab_size,
            ),
            callbacks=callbacks,
            optimizers=(optimizer, None),
            color_consistency_weight=args.color_consistency_weight,
            decoder_context_dropout=args.decoder_context_dropout,
            decoder_context_replacement_token_id=(
                decoder_context_replacement_token_id
            ),
            decoder_context_protected_token_ids=(
                decoder_context_protected_token_ids
            ),
            length_control_enabled=args.length_control_enabled,
            length_pred_weight=args.length_pred_weight,
            length_activate_steps=args.length_activate_steps,
            length_tolerance=args.length_tolerance,
            length_encoder_gradient_scale=args.length_encoder_gradient_scale,
            length_force_confidence=args.length_force_confidence,
            length_ignored_token_ids=(
                model.config.decoder_start_token_id,
                processor.tokenizer.cls_token_id,
                processor.tokenizer.pad_token_id,
                processor.tokenizer.unk_token_id,
            ),
            eos_token_id=processor.tokenizer.sep_token_id,
            bos_token_id=processor.tokenizer.cls_token_id,
            spatial_control_enabled=args.spatial_control_enabled,
            spatial_loss_weight=args.spatial_loss_weight,
        )
        # 设置 trainer 引用，供 forward 包装判断阶段使用
        if args.length_control_enabled:
            model._trainer_ref = trainer

        resume_from_checkpoint = args.resume_from_checkpoint
        if resume_from_checkpoint == "auto":
            resume_from_checkpoint = get_last_checkpoint(str(output_dir))
            print_main(f"自动续训 checkpoint: {resume_from_checkpoint}")
            if resume_from_checkpoint is None:
                raise FileNotFoundError(
                    f"{output_dir} 中没有可续训的 checkpoint-* 目录"
                )
        # 真实数据后训练在第一次更新前记录初始化权重的固定基线。
        # beam 产生空字符串等解码配置故障，与模型权重被训练破坏明确区分开。
        if args.stage == "finetune":
            baseline_metric_key = f"baseline_{args.metric_for_best_model}"
            if not resume_from_checkpoint:
                # EarlyStoppingCallback 固定查找 eval_* 指标。直接使用 baseline_*
                # 会在每个 rank 误报 early stopping 被禁用；先按普通 eval 执行，
                # 再为落盘和阶段收益检查重命名，不改变任何指标数值。
                raw_baseline_metrics = trainer.evaluate()
                baseline_metrics = {
                    (
                        f"baseline_{key.removeprefix('eval_')}"
                        if key.startswith("eval_")
                        else key
                    ): value
                    for key, value in raw_baseline_metrics.items()
                }
                trainer.save_metrics("baseline", baseline_metrics)
                print_main(f"训练前固定验证基线: {baseline_metrics}")
                baseline_loss = baseline_metrics.get("baseline_loss")
                baseline_exact_match = baseline_metrics.get(
                    "baseline_exact_match"
                )
                if (
                    baseline_loss is not None
                    and baseline_exact_match is not None
                    and baseline_exact_match >= 0.50
                    and baseline_loss >= 3.0
                ):
                    raise RuntimeError(
                        "训练前模型完全匹配率较高，但监督 loss 异常偏高："
                        f"exact_match={baseline_exact_match:.2%}, "
                        f"loss={baseline_loss:.4f}。这通常表示 labels、"
                        "decoder_input_ids 与损失函数发生 token 移位错位，"
                        "已在第一次参数更新前停止。"
                    )
                empty_prediction_rate = baseline_metrics.get(
                    "baseline_empty_prediction_rate",
                    0.0,
                )
                if empty_prediction_rate >= 0.50:
                    raise RuntimeError(
                        "训练尚未开始，但验证空预测率已达到 "
                        f"{empty_prediction_rate:.1%}。这是生成解码配置故障，不是"
                        "训练崩盘；请先用 beam=1、length_penalty=1.0、关闭 "
                        "early_stopping 验证长度头置信度与 EOS 约束。"
                    )
            else:
                baseline_path = output_dir / "baseline_results.json"
                if not baseline_path.exists():
                    raise FileNotFoundError(
                        "续训目录缺少 baseline_results.json，无法执行阶段收益检查"
                    )
                baseline_metrics = json.loads(
                    baseline_path.read_text(encoding="utf-8")
                )
                print_main(f"续训沿用固定验证基线: {baseline_metrics}")
            baseline_metric_value = baseline_metrics.get(baseline_metric_key)
            if baseline_metric_value is None:
                raise RuntimeError(
                    f"训练前基线缺少选模指标: {baseline_metric_key}"
                )

        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if baseline_metric_value is not None:
            trained_best_metric = trainer.state.best_metric
            if trained_best_metric is None:
                raise RuntimeError("训练结束但 Trainer 没有记录最佳验证指标")
            tolerance = 1e-6
            did_not_regress = (
                trained_best_metric <= baseline_metric_value + tolerance
                if args.metric_for_best_model == "cer"
                else trained_best_metric + tolerance >= baseline_metric_value
            )
            print_main(
                "阶段收益检查: "
                f"baseline_{args.metric_for_best_model}="
                f"{baseline_metric_value:.6f}, "
                f"trained_best={trained_best_metric:.6f}"
            )
            if not did_not_regress:
                raise RuntimeError(
                    f"{args.stage} 的最佳验证指标仍低于初始化模型；"
                    "拒绝写出退化的 best。请继续使用 "
                    f"{args.cust_data_init_weights_path}。"
                )

        best_dir = output_dir / "best"
        # 标准 DDP 下 Trainer 只让主 rank 写模型；processor 也必须只写一次，
        # 避免多个进程竞争 tokenizer.json / merges.txt。
        trainer.save_model(str(best_dir))
        if is_main_process():
            processor.save_pretrained(str(best_dir))

        if test_dataset is not None:
            test_result = trainer.predict(
                test_dataset,
                metric_key_prefix="test",
            )
            trainer.save_metrics("test", test_result.metrics)
            print_main(f"固定测试集指标: {test_result.metrics}")

        print_main(f"训练完成，最佳模型已保存到: {best_dir.resolve()}")
    finally:
        # 结尾不能再执行 barrier：rank 0 还在写 checkpoint 时，其他 rank 会
        # 卡在 barrier 并持续占用显卡；rank 0 若保存异常则更不会有人解除等待。
        # 直接在每个进程销毁 process group，并在异常路径同样释放模型与缓存。
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            try:
                torch.distributed.destroy_process_group()
            except Exception as cleanup_error:
                print_main(f"警告：分布式进程组释放失败: {cleanup_error}")

        trainer = None
        optimizer = None
        model = None
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                print_main(f"警告：CUDA 缓存释放失败: {cleanup_error}")

        print_main("训练进程组已关闭，CUDA 训练资源已释放。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

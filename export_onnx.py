#!/usr/bin/env python3
"""把印章 TrOCR 导出为两个 ONNX 加一个词表文件。

最终部署目录严格包含：

* ``encoder_model.onnx``：视觉 encoder 与 ``length_control`` 长度预测头；
* ``decoder_model.onnx``：文字 decoder 与预测长度软条件；
* ``vocab.json``：字符词表。

空间辅助头仅用于训练，不进入部署图。图片预处理、生成上限、长度硬约束和
模型配对信息写入 ONNX metadata，因此不再需要额外 config 文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple
import uuid


ENCODER_FILENAME = "encoder_model.onnx"
DECODER_FILENAME = "decoder_model.onnx"
VOCAB_FILENAME = "vocab.json"
BUNDLE_FILENAMES = (
    ENCODER_FILENAME,
    DECODER_FILENAME,
    VOCAB_FILENAME,
)
METADATA_PREFIX = "seal_ocr."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出印章 OCR 三文件 ONNX 部署包"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="训练完成的完整模型目录，通常为 checkpoint_*/best",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="三文件部署包输出目录",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset；默认 17",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--text_confidence_threshold",
        type=float,
        default=0.88,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的三文件部署包；目录含其它文件时仍拒绝",
    )
    parser.add_argument(
        "--skip_runtime_validation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """只允许覆盖本导出器拥有的三个文件，避免误删用户数据。"""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"输出路径不是目录: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return

    entries = {path.name for path in output_dir.iterdir()}
    if not entries:
        return
    unexpected = entries.difference(BUNDLE_FILENAMES)
    if unexpected:
        raise FileExistsError(
            "输出目录含非部署文件，拒绝覆盖: "
            + ", ".join(sorted(unexpected))
        )
    if not overwrite:
        raise FileExistsError(
            f"输出目录已有部署文件: {output_dir}；确认后使用 --overwrite"
        )


def _load_checkpoint_state(torch, model_dir: Path) -> Dict[str, Any]:
    """读取 safetensors/PyTorch 单文件或分片权重。"""
    single_safetensors = model_dir / "model.safetensors"
    single_bin = model_dir / "pytorch_model.bin"
    safetensors_index = model_dir / "model.safetensors.index.json"
    bin_index = model_dir / "pytorch_model.bin.index.json"

    state: Dict[str, Any] = {}
    if single_safetensors.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "模型使用 safetensors，但当前环境没有安装 safetensors"
            ) from exc
        state.update(load_file(str(single_safetensors), device="cpu"))
    elif single_bin.exists():
        state.update(
            torch.load(
                str(single_bin),
                map_location="cpu",
                weights_only=True,
            )
        )
    elif safetensors_index.exists() or bin_index.exists():
        index_path = (
            safetensors_index if safetensors_index.exists() else bin_index
        )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            raise ValueError(f"权重索引没有 weight_map: {index_path}")
        safe_shards = index_path == safetensors_index
        if safe_shards:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError(
                    "模型使用 safetensors，但当前环境没有安装 safetensors"
                ) from exc
        for shard_name in shard_names:
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"权重分片不存在: {shard_path}")
            if safe_shards:
                state.update(load_file(str(shard_path), device="cpu"))
            else:
                state.update(
                    torch.load(
                        str(shard_path),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
    else:
        raise FileNotFoundError(
            f"{model_dir} 中未找到 model.safetensors 或 pytorch_model.bin"
        )

    if any(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    return state


def _attach_length_control(model, state: Mapping[str, Any]):
    """按 checkpoint 参数重建可部署的 length_control v3。"""
    length_state = {
        key.removeprefix("length_control."): value
        for key, value in state.items()
        if key.startswith("length_control.")
    }
    saved_config = getattr(model.config, "length_control_config", {}) or {}
    if not length_state:
        raise RuntimeError(
            "ONNX 导出要求 checkpoint 包含 length_control.* 权重"
        )
    if not bool(saved_config.get("enabled", False)):
        raise RuntimeError(
            "checkpoint 有 length_control 权重，但 config 未声明启用，拒绝猜测"
        )

    architecture_version = int(saved_config.get("architecture_version", 1))
    if architecture_version != 3:
        raise ValueError(
            "ONNX 只支持不依赖真实长度 oracle 的 length_control v3，"
            f"当前 checkpoint 为 v{architecture_version}"
        )

    eos_bias = length_state.get("eos_bias_embedding.weight")
    hidden_bias = length_state.get("hidden_bias_embedding.weight")
    predictor = length_state.get("length_predictor.0.weight")
    if eos_bias is None or hidden_bias is None or predictor is None:
        raise ValueError("length_control 权重不完整，无法推断部署结构")

    from seal_ocr.length_control import LengthConditionModule

    max_target_length = (int(eos_bias.shape[0]) - 1) // 2
    length_control = LengthConditionModule(
        encoder_hidden_size=int(predictor.shape[1]),
        decoder_hidden_size=int(hidden_bias.shape[1]),
        max_target_length=max_target_length,
        predictor_hidden_size=int(predictor.shape[0]),
        predictor_dropout=float(saved_config.get("predictor_dropout", 0.1)),
    )
    length_control.load_state_dict(length_state, strict=True)
    model.length_control = length_control
    return length_control


def _make_export_wrappers(torch, model, length_control, eos_token_id: int):
    """构造 encoder+长度头，以及带软长度条件的 decoder 图。"""

    class EncoderWithLength(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            self.length_control = length_control

        def forward(self, pixel_values):
            outputs = self.encoder(
                pixel_values=pixel_values,
                return_dict=True,
            )
            encoder_hidden_states = outputs.last_hidden_state
            length_logits = self.length_control(encoder_hidden_states)
            return encoder_hidden_states, length_logits

    class DecoderWithLength(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = model.decoder
            self.length_control = length_control
            if self.decoder.get_output_embeddings() is None:
                raise RuntimeError("decoder 没有输出投影，无法导出")
            self.eos_token_id = int(eos_token_id)

        def forward(
            self,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            length_logits,
        ):
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits, _ = self.length_control.apply_length_condition(
                logits=outputs.logits,
                hidden_states=outputs.hidden_states[-1],
                length_logits=length_logits,
                eos_token_id=self.eos_token_id,
                output_projection=self.decoder.get_output_embeddings(),
                current_step=0,
                condition_scale=1.0,
            )
            return logits

    return EncoderWithLength(), DecoderWithLength()


def _float_list(value: Any, default: Sequence[float]) -> list[float]:
    if value is None:
        return [float(item) for item in default]
    if isinstance(value, (int, float)):
        return [float(value)] * len(default)
    values = [float(item) for item in value]
    if len(values) != len(default):
        raise ValueError(f"图片归一化通道数错误: {values}")
    return values


def _required_token_id(name: str, vocab_value: Any, *values: Any) -> int:
    """从配置解析特殊 token，并拒绝词表/配置不一致。"""
    configured = {int(value) for value in values if value is not None}
    if len(configured) > 1:
        raise ValueError(
            f"{name} 在不同模型配置中不一致: {sorted(configured)}"
        )
    resolved = next(iter(configured), int(vocab_value))
    if int(vocab_value) != resolved:
        raise ValueError(
            f"{name} 在 vocab.json 与模型配置中不一致: "
            f"{vocab_value} != {resolved}"
        )
    return resolved


def _deployment_metadata(
    processor,
    model,
    length_control,
    vocab: Mapping[str, int],
    vocab_sha256: str,
    bundle_id: str,
    encoder_hidden_states,
    length_logits,
    opset: int,
    text_confidence_threshold: float,
) -> Dict[str, str]:
    from seal_ocr.image import processor_image_size

    image_width, image_height = processor_image_size(processor)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        image_processor = processor.feature_extractor

    image_mean = _float_list(
        getattr(image_processor, "image_mean", None),
        (0.5, 0.5, 0.5),
    )
    image_std = _float_list(
        getattr(image_processor, "image_std", None),
        (0.5, 0.5, 0.5),
    )
    if any(value == 0.0 for value in image_std):
        raise ValueError(f"图片归一化标准差不能为 0: {image_std}")
    do_rescale = bool(getattr(image_processor, "do_rescale", True))
    do_normalize = bool(getattr(image_processor, "do_normalize", True))
    rescale_factor = float(
        getattr(image_processor, "rescale_factor", 1.0 / 255.0)
    )
    resample = int(getattr(image_processor, "resample", 2))

    generation_config = model.generation_config
    decoder_start_token_id = _required_token_id(
        "decoder_start_token_id",
        vocab["<s>"],
        getattr(generation_config, "decoder_start_token_id", None),
        getattr(model.config, "decoder_start_token_id", None),
    )
    eos_token_id = _required_token_id(
        "eos_token_id",
        vocab["</s>"],
        getattr(generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
    )
    pad_token_id = _required_token_id(
        "pad_token_id",
        vocab["<pad>"],
        getattr(generation_config, "pad_token_id", None),
        getattr(model.config, "pad_token_id", None),
    )
    ignored_token_ids = sorted(
        {
            decoder_start_token_id,
            pad_token_id,
            int(vocab["<unk>"]),
        }
    )
    max_length = getattr(generation_config, "max_length", None)
    if max_length is None:
        max_length = length_control.max_target_length
    length_config = getattr(model.config, "length_control_config", {}) or {}

    return {
        f"{METADATA_PREFIX}bundle_version": "2",
        f"{METADATA_PREFIX}execution_contract": "encoder_decoder_length_v1",
        f"{METADATA_PREFIX}bundle_id": bundle_id,
        f"{METADATA_PREFIX}vocab_file": VOCAB_FILENAME,
        f"{METADATA_PREFIX}vocab_sha256": vocab_sha256,
        f"{METADATA_PREFIX}vocab_size": str(len(vocab)),
        f"{METADATA_PREFIX}image_width": str(image_width),
        f"{METADATA_PREFIX}image_height": str(image_height),
        f"{METADATA_PREFIX}image_resample": str(resample),
        f"{METADATA_PREFIX}resize_mode": str(
            getattr(model.config, "seal_resize_mode", "stretch")
        ),
        f"{METADATA_PREFIX}do_rescale": json.dumps(do_rescale),
        f"{METADATA_PREFIX}rescale_factor": repr(rescale_factor),
        f"{METADATA_PREFIX}do_normalize": json.dumps(do_normalize),
        f"{METADATA_PREFIX}image_mean": json.dumps(image_mean),
        f"{METADATA_PREFIX}image_std": json.dumps(image_std),
        f"{METADATA_PREFIX}max_length": str(int(max_length)),
        f"{METADATA_PREFIX}decoder_start_token_id": str(
            decoder_start_token_id
        ),
        f"{METADATA_PREFIX}eos_token_id": str(eos_token_id),
        f"{METADATA_PREFIX}ignored_token_ids": json.dumps(
            ignored_token_ids
        ),
        f"{METADATA_PREFIX}text_confidence_threshold": repr(
            float(text_confidence_threshold)
        ),
        f"{METADATA_PREFIX}length_control_enabled": "true",
        f"{METADATA_PREFIX}length_force_confidence": repr(
            float(length_config.get("force_confidence", 0.60))
        ),
        f"{METADATA_PREFIX}length_tolerance": str(
            int(length_config.get("tolerance", 0))
        ),
        f"{METADATA_PREFIX}encoder_sequence_length": str(
            int(encoder_hidden_states.shape[1])
        ),
        f"{METADATA_PREFIX}encoder_hidden_size": str(
            int(encoder_hidden_states.shape[2])
        ),
        f"{METADATA_PREFIX}length_class_count": str(
            int(length_logits.shape[1])
        ),
        f"{METADATA_PREFIX}generation_num_beams": str(
            int(getattr(generation_config, "num_beams", 1) or 1)
        ),
        f"{METADATA_PREFIX}generation_no_repeat_ngram_size": str(
            int(
                getattr(
                    generation_config,
                    "no_repeat_ngram_size",
                    0,
                )
                or 0
            )
        ),
        f"{METADATA_PREFIX}generation_do_sample": json.dumps(
            bool(getattr(generation_config, "do_sample", False))
        ),
        f"{METADATA_PREFIX}decoder_mode": "full_sequence_no_kv_cache",
        f"{METADATA_PREFIX}spatial_head_exported": "false",
        f"{METADATA_PREFIX}opset": str(int(opset)),
    }


def _write_onnx_metadata(
    onnx,
    model_path: Path,
    metadata: Mapping[str, str],
    role: str,
) -> None:
    model_proto = onnx.load_model(str(model_path), load_external_data=False)
    merged = {item.key: item.value for item in model_proto.metadata_props}
    merged.update(metadata)
    merged[f"{METADATA_PREFIX}role"] = role
    del model_proto.metadata_props[:]
    for key, value in sorted(merged.items()):
        item = model_proto.metadata_props.add()
        item.key = str(key)
        item.value = str(value)
    onnx.checker.check_model(model_proto)
    onnx.save_model(
        model_proto,
        str(model_path),
        save_as_external_data=False,
    )


def _check_graph_contract(
    onnx,
    model_path: Path,
    expected_inputs: Sequence[str],
    expected_outputs: Sequence[str],
) -> None:
    model_proto = onnx.load_model(str(model_path), load_external_data=False)
    actual_inputs = [value.name for value in model_proto.graph.input]
    actual_outputs = [value.name for value in model_proto.graph.output]
    if actual_inputs != list(expected_inputs):
        raise RuntimeError(
            f"{model_path.name} 输入契约异常: {actual_inputs}"
        )
    if actual_outputs != list(expected_outputs):
        raise RuntimeError(
            f"{model_path.name} 输出契约异常: {actual_outputs}"
        )
    onnx.checker.check_model(model_proto)


def _export_graphs(
    torch,
    encoder_wrapper,
    decoder_wrapper,
    output_dir: Path,
    image_size: Tuple[int, int],
    decoder_start_token_id: int,
    opset: int,
):
    image_width, image_height = image_size
    pixel_values = torch.zeros(
        1,
        3,
        image_height,
        image_width,
        dtype=torch.float32,
    )
    encoder_wrapper.eval()
    decoder_wrapper.eval()
    with torch.no_grad():
        encoder_hidden_states, length_logits = encoder_wrapper(pixel_values)

    encoder_path = output_dir / ENCODER_FILENAME
    torch.onnx.export(
        encoder_wrapper,
        (pixel_values,),
        str(encoder_path),
        input_names=["pixel_values"],
        output_names=["encoder_hidden_states", "length_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "length_logits": {0: "batch_size"},
        },
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )

    input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_args = (
        input_ids,
        attention_mask,
        encoder_hidden_states,
        length_logits,
    )
    with torch.no_grad():
        decoder_logits = decoder_wrapper(*decoder_args)

    decoder_path = output_dir / DECODER_FILENAME
    torch.onnx.export(
        decoder_wrapper,
        decoder_args,
        str(decoder_path),
        input_names=[
            "input_ids",
            "attention_mask",
            "encoder_hidden_states",
            "length_logits",
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
            "attention_mask": {
                0: "batch_size",
                1: "decoder_sequence_length",
            },
            "encoder_hidden_states": {
                0: "batch_size",
                1: "encoder_sequence_length",
            },
            "length_logits": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "decoder_sequence_length"},
        },
        opset_version=opset,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )
    return (
        pixel_values,
        encoder_hidden_states,
        length_logits,
        decoder_args,
        decoder_logits,
    )


def _validate_runtime_outputs(
    torch,
    decoder_wrapper,
    encoder_path: Path,
    decoder_path: Path,
    pixel_values,
    expected_encoder_hidden,
    expected_length_logits,
    decoder_start_token_id: int,
    atol: float,
) -> None:
    """校验 encoder、长度头以及两种 decoder 序列长度。"""
    try:
        import numpy as np
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            "默认导出会使用 ONNX Runtime 校验；请安装 numpy/onnxruntime，"
            "或显式传 --skip_runtime_validation"
        ) from exc

    encoder_session = onnxruntime.InferenceSession(
        str(encoder_path),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = onnxruntime.InferenceSession(
        str(decoder_path),
        providers=["CPUExecutionProvider"],
    )
    encoder_outputs = encoder_session.run(
        ["encoder_hidden_states", "length_logits"],
        {"pixel_values": pixel_values.detach().cpu().numpy()},
    )
    np.testing.assert_allclose(
        encoder_outputs[0],
        expected_encoder_hidden.detach().cpu().numpy(),
        rtol=atol,
        atol=atol,
        err_msg="encoder ONNX 输出与 PyTorch 不一致",
    )
    np.testing.assert_allclose(
        encoder_outputs[1],
        expected_length_logits.detach().cpu().numpy(),
        rtol=atol,
        atol=atol,
        err_msg="length logits ONNX 输出与 PyTorch 不一致",
    )

    for sequence_length in (1, 2):
        input_ids = torch.full(
            (1, sequence_length),
            decoder_start_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            expected_logits = decoder_wrapper(
                input_ids,
                attention_mask,
                expected_encoder_hidden,
                expected_length_logits,
            )
        actual_logits = decoder_session.run(
            ["logits"],
            {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "encoder_hidden_states": encoder_outputs[0],
                "length_logits": encoder_outputs[1],
            },
        )[0]
        np.testing.assert_allclose(
            actual_logits,
            expected_logits.detach().cpu().numpy(),
            rtol=atol,
            atol=atol,
            err_msg=(
                "decoder ONNX 输出与 PyTorch 不一致: "
                f"sequence_length={sequence_length}"
            ),
        )


def export_bundle(args: argparse.Namespace) -> Path:
    if args.opset < 17:
        raise ValueError("长度条件导出要求 ONNX opset >= 17")
    if args.atol <= 0:
        raise ValueError("atol 必须大于 0")
    if not 0.0 <= args.text_confidence_threshold <= 1.0:
        raise ValueError("text_confidence_threshold 必须在 [0, 1] 范围内")

    model_dir = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(f"模型目录不存在: {model_dir}")
    vocab_path = model_dir / VOCAB_FILENAME
    if not vocab_path.is_file():
        raise FileNotFoundError(f"模型目录缺少 vocab.json: {vocab_path}")
    vocab_bytes = vocab_path.read_bytes()
    raw_vocab = json.loads(vocab_bytes.decode("utf-8"))
    if not isinstance(raw_vocab, dict):
        raise ValueError("vocab.json 顶层必须是对象")
    vocab = {str(token): int(token_id) for token, token_id in raw_vocab.items()}
    token_ids = list(vocab.values())
    if len(set(token_ids)) != len(token_ids) or any(
        token_id < 0 for token_id in token_ids
    ):
        raise ValueError("vocab.json 的 token id 必须非负且不能重复")
    for special_token in ("<s>", "</s>", "<pad>", "<unk>"):
        if special_token not in vocab:
            raise ValueError(f"vocab.json 缺少特殊 token: {special_token}")

    _prepare_output_directory(output_dir, args.overwrite)

    try:
        import onnx
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        raise RuntimeError(
            "导出需要 torch、transformers 和 onnx；请先安装训练机依赖"
        ) from exc

    processor = TrOCRProcessor.from_pretrained(str(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
    decoder_vocab_size = int(model.config.decoder.vocab_size)
    if set(token_ids) != set(range(decoder_vocab_size)):
        raise ValueError(
            "vocab.json 的 token id 与 decoder vocab_size 不一致: "
            f"token_count={len(token_ids)}, decoder_vocab_size={decoder_vocab_size}"
        )
    state = _load_checkpoint_state(torch, model_dir)
    length_control = _attach_length_control(model, state)
    del state
    model = model.to(device="cpu", dtype=torch.float32).eval()
    length_control = model.length_control

    generation_config = model.generation_config
    decoder_start_token_id = _required_token_id(
        "decoder_start_token_id",
        vocab["<s>"],
        getattr(generation_config, "decoder_start_token_id", None),
        getattr(model.config, "decoder_start_token_id", None),
    )
    eos_token_id = _required_token_id(
        "eos_token_id",
        vocab["</s>"],
        getattr(generation_config, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
    )
    encoder_wrapper, decoder_wrapper = _make_export_wrappers(
        torch,
        model,
        length_control,
        eos_token_id,
    )

    from seal_ocr.image import processor_image_size

    image_size = processor_image_size(processor)
    vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
    bundle_id = uuid.uuid4().hex

    with tempfile.TemporaryDirectory(
        prefix="seal_ocr_onnx_",
        dir=str(output_dir.parent),
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        (
            pixel_values,
            encoder_hidden_states,
            length_logits,
            _decoder_args,
            _decoder_logits,
        ) = _export_graphs(
            torch,
            encoder_wrapper,
            decoder_wrapper,
            temporary_path,
            image_size,
            decoder_start_token_id,
            args.opset,
        )
        encoder_path = temporary_path / ENCODER_FILENAME
        decoder_path = temporary_path / DECODER_FILENAME

        _check_graph_contract(
            onnx,
            encoder_path,
            ("pixel_values",),
            ("encoder_hidden_states", "length_logits"),
        )
        _check_graph_contract(
            onnx,
            decoder_path,
            (
                "input_ids",
                "attention_mask",
                "encoder_hidden_states",
                "length_logits",
            ),
            ("logits",),
        )
        metadata = _deployment_metadata(
            processor,
            model,
            length_control,
            vocab,
            vocab_sha256,
            bundle_id,
            encoder_hidden_states,
            length_logits,
            args.opset,
            args.text_confidence_threshold,
        )
        _write_onnx_metadata(onnx, encoder_path, metadata, "encoder")
        _write_onnx_metadata(onnx, decoder_path, metadata, "decoder")
        shutil.copyfile(vocab_path, temporary_path / VOCAB_FILENAME)

        if not args.skip_runtime_validation:
            _validate_runtime_outputs(
                torch,
                decoder_wrapper,
                encoder_path,
                decoder_path,
                pixel_values,
                encoder_hidden_states,
                length_logits,
                decoder_start_token_id,
                args.atol,
            )

        generated = {path.name for path in temporary_path.iterdir()}
        if generated != set(BUNDLE_FILENAMES):
            raise RuntimeError(
                "导出结果不符合三文件约定: "
                + ", ".join(sorted(generated))
            )
        for filename in BUNDLE_FILENAMES:
            os.replace(temporary_path / filename, output_dir / filename)

    final_files = {path.name for path in output_dir.iterdir()}
    if final_files != set(BUNDLE_FILENAMES):
        raise RuntimeError(
            "最终部署目录不是严格三文件结构: "
            + ", ".join(sorted(final_files))
        )
    return output_dir


def main() -> int:
    args = build_parser().parse_args()
    output_dir = export_bundle(args)
    print(f"三文件 ONNX 部署包已生成: {output_dir}")
    for filename in BUNDLE_FILENAMES:
        path = output_dir / filename
        print(f"- {filename}: {path.stat().st_size / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

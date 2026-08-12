#!/usr/bin/env python3
"""三文件 ONNX 部署包的参考推理实现。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from typing import Mapping, Optional, Sequence


METADATA_PREFIX = "seal_ocr."
ENCODER_FILENAME = "encoder_model.onnx"
DECODER_FILENAME = "decoder_model.onnx"
EXPECTED_ENCODER_INPUTS = {"pixel_values"}
EXPECTED_ENCODER_OUTPUTS = {"encoder_hidden_states", "length_logits"}
EXPECTED_DECODER_INPUTS = {
    "input_ids",
    "attention_mask",
    "encoder_hidden_states",
    "length_logits",
}
EXPECTED_DECODER_OUTPUTS = {"logits"}


def _metadata_value(
    metadata: Mapping[str, str],
    key: str,
    default: Optional[str] = None,
) -> str:
    full_key = f"{METADATA_PREFIX}{key}"
    if full_key in metadata:
        return metadata[full_key]
    if default is not None:
        return default
    raise ValueError(f"ONNX metadata 缺少 {full_key}")


def _metadata_bool(
    metadata: Mapping[str, str],
    key: str,
    default: Optional[bool] = None,
) -> bool:
    fallback = None if default is None else json.dumps(default)
    value = _metadata_value(metadata, key, fallback)
    parsed = json.loads(value)
    if not isinstance(parsed, bool):
        raise ValueError(f"ONNX metadata {key} 不是布尔值: {value!r}")
    return parsed


def validate_metadata_pair(
    encoder_metadata: Mapping[str, str],
    decoder_metadata: Mapping[str, str],
) -> dict[str, str]:
    """确保 encoder/decoder 来自同一次导出且配置完全一致。"""
    encoder = {
        key: value
        for key, value in encoder_metadata.items()
        if key.startswith(METADATA_PREFIX)
    }
    decoder = {
        key: value
        for key, value in decoder_metadata.items()
        if key.startswith(METADATA_PREFIX)
    }
    if _metadata_value(encoder, "role") != "encoder":
        raise ValueError("encoder_model.onnx 的 role metadata 不正确")
    if _metadata_value(decoder, "role") != "decoder":
        raise ValueError("decoder_model.onnx 的 role metadata 不正确")

    role_key = f"{METADATA_PREFIX}role"
    encoder.pop(role_key, None)
    decoder.pop(role_key, None)
    if encoder != decoder:
        differing_keys = sorted(
            key
            for key in set(encoder).union(decoder)
            if encoder.get(key) != decoder.get(key)
        )
        raise ValueError(
            "encoder/decoder metadata 不一致，可能混用了不同部署包: "
            + ", ".join(differing_keys)
        )
    return encoder


@dataclass(frozen=True)
class DeploymentConfig:
    bundle_id: str
    vocab_file: str
    vocab_sha256: str
    vocab_size: int
    image_width: int
    image_height: int
    image_resample: int
    resize_mode: str
    do_rescale: bool
    rescale_factor: float
    do_normalize: bool
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    max_length: int
    decoder_start_token_id: int
    eos_token_id: int
    ignored_token_ids: tuple[int, ...]
    text_confidence_threshold: float
    length_force_confidence: float
    length_tolerance: int
    generation_num_beams: int
    generation_no_repeat_ngram_size: int
    generation_do_sample: bool

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
    ) -> "DeploymentConfig":
        bundle_version = int(_metadata_value(metadata, "bundle_version"))
        contract = _metadata_value(metadata, "execution_contract")
        if bundle_version != 2 or contract != "encoder_decoder_length_v1":
            raise ValueError(
                "不支持的 ONNX 部署契约: "
                f"bundle_version={bundle_version}, contract={contract!r}"
            )
        if not _metadata_bool(metadata, "length_control_enabled"):
            raise ValueError("当前部署契约要求启用 length_control")

        image_mean = tuple(
            float(value)
            for value in json.loads(_metadata_value(metadata, "image_mean"))
        )
        image_std = tuple(
            float(value)
            for value in json.loads(_metadata_value(metadata, "image_std"))
        )
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("ONNX metadata 的图片均值/标准差必须为 3 通道")

        config = cls(
            bundle_id=_metadata_value(metadata, "bundle_id"),
            vocab_file=_metadata_value(metadata, "vocab_file"),
            vocab_sha256=_metadata_value(metadata, "vocab_sha256"),
            vocab_size=int(_metadata_value(metadata, "vocab_size")),
            image_width=int(_metadata_value(metadata, "image_width")),
            image_height=int(_metadata_value(metadata, "image_height")),
            image_resample=int(_metadata_value(metadata, "image_resample")),
            resize_mode=_metadata_value(metadata, "resize_mode"),
            do_rescale=_metadata_bool(metadata, "do_rescale"),
            rescale_factor=float(
                _metadata_value(metadata, "rescale_factor")
            ),
            do_normalize=_metadata_bool(metadata, "do_normalize"),
            image_mean=image_mean,
            image_std=image_std,
            max_length=int(_metadata_value(metadata, "max_length")),
            decoder_start_token_id=int(
                _metadata_value(metadata, "decoder_start_token_id")
            ),
            eos_token_id=int(_metadata_value(metadata, "eos_token_id")),
            ignored_token_ids=tuple(
                int(value)
                for value in json.loads(
                    _metadata_value(metadata, "ignored_token_ids")
                )
            ),
            text_confidence_threshold=float(
                _metadata_value(metadata, "text_confidence_threshold")
            ),
            length_force_confidence=float(
                _metadata_value(metadata, "length_force_confidence")
            ),
            length_tolerance=int(
                _metadata_value(metadata, "length_tolerance")
            ),
            generation_num_beams=int(
                _metadata_value(metadata, "generation_num_beams", "1")
            ),
            generation_no_repeat_ngram_size=int(
                _metadata_value(
                    metadata,
                    "generation_no_repeat_ngram_size",
                    "0",
                )
            ),
            generation_do_sample=_metadata_bool(
                metadata,
                "generation_do_sample",
                False,
            ),
        )
        if not config.bundle_id:
            raise ValueError("bundle_id 不能为空")
        vocab_path = Path(config.vocab_file)
        if (
            vocab_path.parts != (config.vocab_file,)
            or config.vocab_file in {"", ".", ".."}
        ):
            raise ValueError("vocab_file 必须是部署目录内的文件名")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0")
        if config.resize_mode not in {"stretch", "letterbox"}:
            raise ValueError(
                f"不支持的 resize_mode: {config.resize_mode!r}"
            )
        if config.image_width <= 0 or config.image_height <= 0:
            raise ValueError("ONNX metadata 中的图片尺寸必须大于 0")
        if config.max_length < 2:
            raise ValueError("ONNX metadata 中的 max_length 必须至少为 2")
        if config.length_tolerance < 0:
            raise ValueError("length_tolerance 不能为负数")
        if any(value == 0.0 for value in config.image_std):
            raise ValueError("图片归一化标准差不能为 0")
        for name, value in (
            ("text_confidence_threshold", config.text_confidence_threshold),
            ("length_force_confidence", config.length_force_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1] 范围内")
        if (
            config.generation_num_beams != 1
            or config.generation_no_repeat_ngram_size != 0
            or config.generation_do_sample
        ):
            raise ValueError(
                "infer_onnx.py 只支持当前训练默认的 greedy 解码："
                "num_beams=1、no_repeat_ngram_size=0、do_sample=false"
            )
        return config


def read_vocab_file(
    model_dir: Path,
    config: DeploymentConfig,
) -> dict[str, int]:
    vocab_path = model_dir / config.vocab_file
    if not vocab_path.is_file():
        raise FileNotFoundError(f"部署包缺少词表: {vocab_path}")
    vocab_bytes = vocab_path.read_bytes()
    actual_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
    if actual_sha256 != config.vocab_sha256:
        raise ValueError(
            "vocab.json 摘要不一致，部署包可能损坏或混用了文件: "
            f"expected={config.vocab_sha256}, actual={actual_sha256}"
        )
    vocab = json.loads(vocab_bytes.decode("utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError("vocab.json 顶层不是对象")
    normalized = {str(token): int(token_id) for token, token_id in vocab.items()}
    if len(normalized) != config.vocab_size:
        raise ValueError(
            "vocab.json 大小与 ONNX metadata 不一致: "
            f"{len(normalized)} != {config.vocab_size}"
        )
    for special_token in ("<s>", "</s>", "<pad>", "<unk>"):
        if special_token not in normalized:
            raise ValueError(f"vocab.json 缺少 {special_token}")
    token_ids = list(normalized.values())
    if set(token_ids) != set(range(config.vocab_size)):
        raise ValueError(
            "vocab.json 的 token id 必须从 0 连续到 vocab_size - 1"
        )
    if normalized["<s>"] != config.decoder_start_token_id:
        raise ValueError("vocab.json 的 <s> 与 decoder_start_token_id 不一致")
    if normalized["</s>"] != config.eos_token_id:
        raise ValueError("vocab.json 的 </s> 与 eos_token_id 不一致")
    return normalized


def decode_text(
    tokens: Sequence[int],
    vocab: Mapping[str, int],
) -> str:
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    ignored = {
        vocab.get("<s>"),
        vocab.get("<pad>"),
        vocab.get("<unk>"),
    }
    eos_token_id = vocab["</s>"]
    text = []
    for token_id in tokens:
        if token_id == eos_token_id:
            break
        if token_id not in ignored:
            text.append(inverse_vocab.get(int(token_id), ""))
    return "".join(text)


def length_constraint_action(
    token_ids: Sequence[int],
    predicted_length: int,
    confidence: float,
    minimum_confidence: float,
    tolerance: int,
    eos_token_id: int,
    ignored_token_ids: Sequence[int],
) -> str:
    """返回 ``none``、``block_eos`` 或 ``force_eos``。"""
    if confidence < minimum_confidence:
        return "none"
    ignored = set(int(value) for value in ignored_token_ids)
    generated_length = sum(
        1
        for token_id in token_ids
        if int(token_id) != eos_token_id and int(token_id) not in ignored
    )
    if generated_length < int(predicted_length) - int(tolerance):
        return "block_eos"
    if generated_length >= int(predicted_length) + int(tolerance):
        return "force_eos"
    return "none"


def _border_background_color(image) -> tuple[int, int, int]:
    from PIL import Image, ImageDraw, ImageStat

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


def prepare_pixel_values(image, config: DeploymentConfig):
    import numpy as np
    from PIL import Image

    image = image.convert("RGB")
    try:
        processor_resample = Image.Resampling(config.image_resample)
    except ValueError as exc:
        raise ValueError(
            f"ONNX metadata 中的 image_resample 无效: {config.image_resample}"
        ) from exc

    if config.resize_mode == "letterbox":
        source_width, source_height = image.size
        scale = min(
            config.image_width / source_width,
            config.image_height / source_height,
        )
        resized_width = max(
            1,
            min(config.image_width, round(source_width * scale)),
        )
        resized_height = max(
            1,
            min(config.image_height, round(source_height * scale)),
        )
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.LANCZOS,
        )
        prepared = Image.new(
            "RGB",
            (config.image_width, config.image_height),
            _border_background_color(image),
        )
        prepared.paste(
            resized,
            (
                (config.image_width - resized_width) // 2,
                (config.image_height - resized_height) // 2,
            ),
        )
    else:
        prepared = image.resize(
            (config.image_width, config.image_height),
            resample=processor_resample,
        )

    pixel_values = np.asarray(prepared, dtype=np.float32)
    if config.do_rescale:
        pixel_values = pixel_values * config.rescale_factor
    if config.do_normalize:
        mean = np.asarray(config.image_mean, dtype=np.float32)
        std = np.asarray(config.image_std, dtype=np.float32)
        pixel_values = (pixel_values - mean) / std
    return np.transpose(pixel_values, (2, 0, 1))[None, ...]


def _softmax(values):
    import numpy as np

    finite = np.isfinite(values)
    if not bool(finite.any()):
        raise RuntimeError("decoder 返回的 token logits 全部不是有限值")
    shifted = values - np.max(values[finite])
    exponentials = np.exp(shifted)
    denominator = exponentials.sum()
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("decoder token logits 无法计算 softmax")
    return exponentials / denominator


class OnnxSealOcr:
    def __init__(
        self,
        model_path: str,
        provider: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        import numpy as np
        import onnxruntime

        model_dir = Path(model_path).expanduser()
        if model_dir.is_file() and model_dir.name in {
            ENCODER_FILENAME,
            DECODER_FILENAME,
        }:
            model_dir = model_dir.parent
        if not model_dir.is_dir():
            raise NotADirectoryError(f"ONNX 部署包目录不存在: {model_dir}")

        for filename in (
            ENCODER_FILENAME,
            DECODER_FILENAME,
            "vocab.json",
        ):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"ONNX 部署包缺少文件: {filename}")

        providers = [provider] if provider else ["CPUExecutionProvider"]
        self.encoder = onnxruntime.InferenceSession(
            str(model_dir / ENCODER_FILENAME),
            providers=providers,
        )
        self.decoder = onnxruntime.InferenceSession(
            str(model_dir / DECODER_FILENAME),
            providers=providers,
        )
        self._check_session_contract(
            self.encoder,
            EXPECTED_ENCODER_INPUTS,
            EXPECTED_ENCODER_OUTPUTS,
            ENCODER_FILENAME,
        )
        self._check_session_contract(
            self.decoder,
            EXPECTED_DECODER_INPUTS,
            EXPECTED_DECODER_OUTPUTS,
            DECODER_FILENAME,
        )

        metadata = validate_metadata_pair(
            self.encoder.get_modelmeta().custom_metadata_map,
            self.decoder.get_modelmeta().custom_metadata_map,
        )
        self.config = DeploymentConfig.from_metadata(metadata)
        self.vocab = read_vocab_file(model_dir, self.config)
        self.confidence_threshold = (
            self.config.text_confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必须在 [0, 1] 范围内")
        self._np = np

    @staticmethod
    def _check_session_contract(
        session,
        expected_inputs: set[str],
        expected_outputs: set[str],
        filename: str,
    ) -> None:
        input_names = {value.name for value in session.get_inputs()}
        output_names = {value.name for value in session.get_outputs()}
        if input_names != expected_inputs:
            raise ValueError(
                f"{filename} 输入契约不一致: "
                + ", ".join(sorted(input_names))
            )
        if output_names != expected_outputs:
            raise ValueError(
                f"{filename} 输出契约不一致: "
                + ", ".join(sorted(output_names))
            )

    def run(self, image) -> str:
        np = self._np
        pixel_values = prepare_pixel_values(image, self.config)
        encoder_hidden_states, length_logits = self.encoder.run(
            ["encoder_hidden_states", "length_logits"],
            {"pixel_values": pixel_values},
        )
        length_probabilities = _softmax(length_logits[0])
        predicted_length = int(length_probabilities.argmax())
        length_confidence = float(length_probabilities[predicted_length])

        ids = [self.config.decoder_start_token_id]
        scores: list[float] = []
        while len(ids) < self.config.max_length:
            input_ids = np.asarray([ids], dtype=np.int64)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            logits = self.decoder.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "length_logits": length_logits,
                },
            )[0]
            token_logits = logits[0, -1].astype(np.float64, copy=True)
            action = length_constraint_action(
                ids,
                predicted_length,
                length_confidence,
                self.config.length_force_confidence,
                self.config.length_tolerance,
                self.config.eos_token_id,
                self.config.ignored_token_ids,
            )
            if action == "block_eos":
                token_logits[self.config.eos_token_id] = -np.inf
            elif action == "force_eos":
                token_logits.fill(-np.inf)
                token_logits[self.config.eos_token_id] = 0.0

            probabilities = _softmax(token_logits)
            next_token = int(probabilities.argmax())
            if next_token == self.config.eos_token_id:
                break
            scores.append(float(probabilities[next_token]))
            ids.append(next_token)

        average_confidence = statistics.fmean(scores) if scores else 0.0
        if average_confidence <= self.confidence_threshold:
            return ""
        return decode_text(ids, self.vocab)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试三文件印章 OCR ONNX")
    parser.add_argument(
        "--model",
        required=True,
        help="包含两个 ONNX 和 vocab.json 的部署包目录",
    )
    parser.add_argument("--image", "--test_img", dest="test_img", required=True, help="测试图片")
    parser.add_argument(
        "--provider",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    from PIL import Image

    args = build_parser().parse_args()
    model = OnnxSealOcr(
        args.model,
        provider=args.provider,
        confidence_threshold=args.confidence_threshold,
    )
    with Image.open(args.test_img) as image:
        result = model.run(image)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

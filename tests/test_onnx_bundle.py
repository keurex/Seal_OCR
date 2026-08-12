"""三文件 ONNX 部署契约的纯 Python 回归测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from export_onnx import (
    BUNDLE_FILENAMES,
    DECODER_FILENAME,
    ENCODER_FILENAME,
    VOCAB_FILENAME,
    _prepare_output_directory,
)
from infer_onnx import (
    DeploymentConfig,
    decode_text,
    length_constraint_action,
    read_vocab_file,
    validate_metadata_pair,
)


def _vocab_bytes() -> bytes:
    return json.dumps(
        {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
            "测": 4,
            "试": 5,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _metadata(role: str) -> dict[str, str]:
    vocab_bytes = _vocab_bytes()
    prefix = "seal_ocr."
    return {
        f"{prefix}bundle_version": "2",
        f"{prefix}execution_contract": "encoder_decoder_length_v1",
        f"{prefix}bundle_id": "test-bundle-id",
        f"{prefix}role": role,
        f"{prefix}vocab_file": "vocab.json",
        f"{prefix}vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        f"{prefix}vocab_size": "6",
        f"{prefix}image_width": "512",
        f"{prefix}image_height": "512",
        f"{prefix}image_resample": "2",
        f"{prefix}resize_mode": "letterbox",
        f"{prefix}do_rescale": "true",
        f"{prefix}rescale_factor": repr(1.0 / 255.0),
        f"{prefix}do_normalize": "true",
        f"{prefix}image_mean": "[0.5, 0.5, 0.5]",
        f"{prefix}image_std": "[0.5, 0.5, 0.5]",
        f"{prefix}max_length": "40",
        f"{prefix}decoder_start_token_id": "0",
        f"{prefix}eos_token_id": "2",
        f"{prefix}ignored_token_ids": "[0, 1, 3]",
        f"{prefix}text_confidence_threshold": "0.88",
        f"{prefix}length_control_enabled": "true",
        f"{prefix}length_force_confidence": "0.9",
        f"{prefix}length_tolerance": "0",
        f"{prefix}generation_num_beams": "1",
        f"{prefix}generation_no_repeat_ngram_size": "0",
        f"{prefix}generation_do_sample": "false",
    }


class OnnxBundleTest(unittest.TestCase):
    def test_bundle_contract_contains_exactly_three_files(self) -> None:
        self.assertEqual(
            BUNDLE_FILENAMES,
            (
                "encoder_model.onnx",
                "decoder_model.onnx",
                "vocab.json",
            ),
        )
        self.assertEqual(ENCODER_FILENAME, "encoder_model.onnx")
        self.assertEqual(DECODER_FILENAME, "decoder_model.onnx")
        self.assertEqual(VOCAB_FILENAME, "vocab.json")

    def test_metadata_pair_and_vocab_are_consistent(self) -> None:
        metadata = validate_metadata_pair(
            _metadata("encoder"),
            _metadata("decoder"),
        )
        config = DeploymentConfig.from_metadata(metadata)
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir)
            (model_dir / "vocab.json").write_bytes(_vocab_bytes())
            vocab = read_vocab_file(model_dir, config)

        self.assertEqual((config.image_width, config.image_height), (512, 512))
        self.assertEqual(config.resize_mode, "letterbox")
        self.assertEqual(decode_text([0, 0, 4, 5, 2], vocab), "测试")

    def test_mixed_encoder_and_decoder_are_rejected(self) -> None:
        decoder_metadata = _metadata("decoder")
        decoder_metadata["seal_ocr.bundle_id"] = "another-export"
        with self.assertRaisesRegex(ValueError, "混用了不同部署包"):
            validate_metadata_pair(
                _metadata("encoder"),
                decoder_metadata,
            )

    def test_vocab_corruption_is_rejected(self) -> None:
        metadata = validate_metadata_pair(
            _metadata("encoder"),
            _metadata("decoder"),
        )
        config = DeploymentConfig.from_metadata(metadata)
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir)
            (model_dir / "vocab.json").write_bytes(_vocab_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "摘要不一致"):
                read_vocab_file(model_dir, config)

    def test_length_constraint_matches_training_visible_token_count(self) -> None:
        common = {
            "predicted_length": 2,
            "confidence": 0.95,
            "minimum_confidence": 0.90,
            "tolerance": 0,
            "eos_token_id": 2,
            "ignored_token_ids": (0, 1, 3),
        }
        self.assertEqual(
            length_constraint_action([0, 0], **common),
            "block_eos",
        )
        self.assertEqual(
            length_constraint_action([0, 0, 4], **common),
            "block_eos",
        )
        self.assertEqual(
            length_constraint_action([0, 0, 4, 5], **common),
            "force_eos",
        )
        self.assertEqual(
            length_constraint_action(
                [0, 0],
                **{**common, "confidence": 0.20},
            ),
            "none",
        )

    def test_overwrite_refuses_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "model"
            output_dir.mkdir()
            (output_dir / "notes.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "非部署文件"):
                _prepare_output_directory(output_dir, overwrite=True)

    def test_overwrite_accepts_only_owned_deployment_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "model"
            output_dir.mkdir()
            for filename in BUNDLE_FILENAMES:
                (output_dir / filename).write_bytes(b"old")
            _prepare_output_directory(output_dir, overwrite=True)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                set(BUNDLE_FILENAMES),
            )


if __name__ == "__main__":
    unittest.main()

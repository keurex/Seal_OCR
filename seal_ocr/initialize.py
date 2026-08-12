"""用公司名字符词表初始化 TrOCR 模型。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# 初始化只在 CPU 上执行，避免占用正式训练 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def _square_grid_size(token_count: int) -> int | None:
    """返回平方 patch 网格边长；不是完整平方数时返回 None。"""
    if token_count <= 0:
        return None
    side = int(token_count**0.5)
    return side if side * side == token_count else None


def _resize_vision_position_embeddings(torch, source, target):
    """用双三次插值把 ViT/DeiT 的二维位置编码迁移到新输入分辨率。

    DeiT 有 CLS 和 distillation 两个前缀 token，ViT 通常只有 CLS。这里同时
    兼容两种结构，只插值 patch 网格，前缀 token 原样复用。
    """
    if (
        source.ndim != 3
        or target.ndim != 3
        or source.shape[0] != 1
        or target.shape[0] != 1
        or source.shape[2] != target.shape[2]
    ):
        raise ValueError(
            "视觉位置编码尺寸不兼容: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )

    grid_spec = None
    for prefix_tokens in (2, 1, 0):
        source_side = _square_grid_size(source.shape[1] - prefix_tokens)
        target_side = _square_grid_size(target.shape[1] - prefix_tokens)
        if source_side is not None and target_side is not None:
            grid_spec = (prefix_tokens, source_side, target_side)
            break
    if grid_spec is None:
        raise ValueError(
            "无法从视觉位置编码推断二维 patch 网格: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )

    prefix_tokens, source_side, target_side = grid_spec
    prefix = source[:, :prefix_tokens]
    patch_positions = source[:, prefix_tokens:]
    original_dtype = patch_positions.dtype
    patch_positions = patch_positions.float().reshape(
        1,
        source_side,
        source_side,
        source.shape[2],
    )
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = torch.nn.functional.interpolate(
        patch_positions,
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1,
        target_side * target_side,
        target.shape[2],
    )
    resized = torch.cat([prefix.float(), patch_positions], dim=1)
    return resized.to(dtype=original_dtype)


def _set_processor_image_size(preprocessor_path: Path, image_size: int) -> None:
    """同步保存后的 image processor 尺寸，保证训练/评测/部署一致。"""
    config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    config["size"] = {"height": image_size, "width": image_size}
    if "crop_size" in config:
        config["crop_size"] = {"height": image_size, "width": image_size}
    preprocessor_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_custom_tokenizer(custom_vocab: dict, output_path: str) -> None:
    """直接修改 tokenizer.json 的 model.vocab，使 tokenizer 与 model decoder 词表对齐。

    tokenizers 库的 add_tokens 只追加到 added_tokens，不影响 base BPE vocab，
    导致 vocab_size=0。正确做法是直接覆写 tokenizer.json 中的 model.vocab 字段。
    """
    import json as _json

    tokenizer_json_path = Path(output_path) / "tokenizer.json"
    if tokenizer_json_path.exists():
        tok_data = _json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    else:
        tok_data = {"version": "1.0", "truncation": None, "padding": None}

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
        _json.dumps(tok_data, ensure_ascii=False),
        encoding="utf-8",
    )

    # 确保 merges.txt 存在（RobertaTokenizerFast 必需）
    (Path(output_path) / "merges.txt").write_text(
        "#version: 0.2\n", encoding="utf-8"
    )

    # 更新 tokenizer_config.json
    config_path = Path(output_path) / "tokenizer_config.json"
    cfg = {}
    if config_path.exists():
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "RobertaTokenizer"
    cfg["model_max_length"] = 512
    config_path.write_text(
        _json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"自定义 tokenizer 已生成：vocab_size = {len(custom_vocab)}")


def read_vocab(vocab_path: str):
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for line in Path(vocab_path).read_text(encoding="utf-8").splitlines():
        token = line.strip("\n")
        if token and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="初始化公司名 TrOCR 字符模型")
    parser.add_argument(
        "--cust_vocab",
        default="./models/vocab.txt",
        help="逐行字符词表",
    )
    parser.add_argument(
        "--pretrain_model",
        default="./models/base-trocr",
        help="基础 TrOCR 权重",
    )
    parser.add_argument(
        "--cust_data_init_weights_path",
        default="./models/init",
        help="新的初始化模型输出目录",
    )
    parser.add_argument(
        "--allow_unknown_chars",
        action="store_true",
        help="显式允许基础模型没有的字符用 <unk> 权重初始化",
    )
    parser.add_argument(
        "--allow_existing_output",
        action="store_true",
        help="显式允许写入非空输出目录；建议始终使用新版本目录",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help=(
            "可选的新方形输入边长。设置为 512 时会插值 DeiT 二维位置编码，"
            "保留其余预训练权重"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from transformers import AutoConfig, TrOCRProcessor, VisionEncoderDecoderModel

    output_path = Path(args.cust_data_init_weights_path).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        if not args.allow_existing_output:
            raise FileExistsError(
                f"输出目录非空: {output_path}。请换新目录，"
                "或确认后使用 --allow_existing_output。"
            )
    output_path.mkdir(parents=True, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(args.pretrain_model)
    base_model = VisionEncoderDecoderModel.from_pretrained(
        args.pretrain_model,
        ignore_mismatched_sizes=False,
    )
    base_image_size = int(base_model.config.encoder.image_size)
    patch_size = int(base_model.config.encoder.patch_size)
    image_size = args.image_size or base_image_size
    if image_size < patch_size or image_size % patch_size != 0:
        raise ValueError(
            f"image_size={image_size} 必须不小于 patch_size={patch_size}，"
            "且能被 patch_size 整除"
        )
    base_vocab = processor.tokenizer.get_vocab()
    custom_vocab = read_vocab(args.cust_vocab)

    missing_tokens = [
        token
        for token in custom_vocab
        if token not in base_vocab and token not in SPECIAL_TOKENS
    ]
    if missing_tokens and not args.allow_unknown_chars:
        raise ValueError(
            f"基础模型词表缺少 {len(missing_tokens)} 个字符: "
            f"{missing_tokens[:50]}。优先更换覆盖这些字符的基础模型；"
            "确认接受这些字符用随机初始化独立 embedding 时再使用 --allow_unknown_chars。"
        )

    unknown_index = base_vocab["<unk>"]
    # 基础词表中存在的字符：直接复用对应 embedding 行；
    # 缺失字符分配负数占位，后续 index_select 时用随机初始化补齐，确保独立 embedding。
    extended_base_size = len(base_vocab)
    keep_tokens = []
    missing_indices = []  # custom_vocab 中缺失字符的位置
    for position, token in enumerate(custom_vocab):
        if token in base_vocab:
            keep_tokens.append(base_vocab[token])
        else:
            keep_tokens.append(extended_base_size)  # 占位，指向 base 之外的新行
            missing_indices.append(position)

    processor.save_pretrained(str(output_path))
    base_model.save_pretrained(str(output_path))
    _set_processor_image_size(output_path / "preprocessor_config.json", image_size)
    (output_path / "vocab.json").write_text(
        json.dumps(custom_vocab, ensure_ascii=False),
        encoding="utf-8",
    )

    # 用自定义词表覆盖 tokenizer.json，保证 tokenizer.vocab_size 与模型 decoder.vocab_size 对齐
    _write_custom_tokenizer(custom_vocab, str(output_path))

    config_path = output_path / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config["decoder"]["vocab_size"] = len(custom_vocab)
    model_config["vocab_size"] = len(custom_vocab)
    model_config["encoder"]["image_size"] = image_size
    config_path.write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    custom_config = AutoConfig.from_pretrained(str(output_path))
    custom_model = VisionEncoderDecoderModel(custom_config)
    base_state = base_model.state_dict()
    custom_state = custom_model.state_dict()

    # 词表 resize：base 中已有的字符复用 embedding；缺失字符用随机初始化补齐。
    resized_keys = []
    init_std = float(
        getattr(custom_config.decoder, "init_std", 0.02)
    )
    for key, target_value in custom_state.items():
        source_value = base_state.get(key)
        if source_value is None:
            raise KeyError(f"基础模型缺少参数: {key}")
        if source_value.shape == target_value.shape:
            custom_state[key] = source_value
            continue
        can_resize_vocab_axis = (
            source_value.ndim >= 1
            and target_value.ndim == source_value.ndim
            and source_value.shape[0] == len(base_vocab)
            and target_value.shape[0] == len(custom_vocab)
            and source_value.shape[1:] == target_value.shape[1:]
        )
        can_resize_position_embeddings = (
            "position_embeddings" in key
            and source_value.ndim == 3
            and target_value.ndim == 3
            and source_value.shape[0] == target_value.shape[0] == 1
            and source_value.shape[2] == target_value.shape[2]
        )
        if can_resize_position_embeddings:
            custom_state[key] = _resize_vision_position_embeddings(
                torch,
                source_value,
                target_value,
            )
            resized_keys.append(key)
            continue
        if not can_resize_vocab_axis:
            raise ValueError(
                f"不可迁移的参数尺寸不一致: {key}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        # 先用随机初始化填充全部新词表行，再覆盖 base 中已有的行
        new_tensor = torch.randn_like(target_value) * init_std
        keep_token_indices = []
        keep_target_positions = []
        for position, base_idx in enumerate(keep_tokens):
            if base_idx < len(base_vocab):
                keep_token_indices.append(base_idx)
                keep_target_positions.append(position)
        if keep_token_indices:
            new_tensor.index_copy_(
                0,
                torch.tensor(keep_target_positions, dtype=torch.long),
                source_value.index_select(
                    0,
                    torch.tensor(keep_token_indices, dtype=torch.long),
                ),
            )
        custom_state[key] = new_tensor
        resized_keys.append(key)

    custom_model.load_state_dict(custom_state)

    # 基础 TrOCR 权重自带的生成配置偏向通用文本（length_penalty=2、
    # no_repeat_ngram_size=3）。公司名 OCR 使用确定的字符序列，初始化产物就改成
    # 与训练/评测一致的配置，避免后续脚本意外继承旧语言生成先验。
    generation_values = {
        "bos_token_id": custom_vocab["<s>"],
        "decoder_start_token_id": custom_vocab["<s>"],
        "pad_token_id": custom_vocab["<pad>"],
        "eos_token_id": custom_vocab["</s>"],
        "max_length": 40,
        "min_new_tokens": 4,
        "num_beams": 1,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
    }
    for key, value in generation_values.items():
        setattr(custom_model.generation_config, key, value)

    # 只有 forward/shift_tokens_right 必需的特殊 token 保留在 model config；
    # beam、长度等生成参数只存 generation_config，避免训练保存时迁移告警。
    for key in (
        "bos_token_id",
        "decoder_start_token_id",
        "pad_token_id",
        "eos_token_id",
    ):
        setattr(custom_model.config, key, generation_values[key])
        setattr(custom_model.config.decoder, key, generation_values[key])
        setattr(custom_model.decoder.config, key, generation_values[key])
    for key in (
        "max_length",
        "min_new_tokens",
        "num_beams",
        "length_penalty",
        "no_repeat_ngram_size",
        "early_stopping",
    ):
        setattr(custom_model.config, key, None)

    # train.py 的 collator 已把含 BOS/EOS 的 labels 右移为 decoder 输入，
    # 初始化配置必须保存未移位 token CE；ForCausalLM 会把监督目标再次移动。
    custom_model.loss_type = "ForMaskedLM"
    custom_model.config.loss_type = "ForMaskedLM"
    custom_model.config.seal_resize_mode = "letterbox"

    custom_model.save_pretrained(str(output_path))
    parameter_count = sum(
        parameter.numel() for parameter in custom_model.parameters()
    )
    print(f"基础词表: {len(base_vocab)}")
    print(f"公司名词表（含特殊 token）: {len(custom_vocab)}")
    print(
        f"视觉输入: {base_image_size} -> {image_size}，"
        f"patch 网格: {base_image_size // patch_size}×{base_image_size // patch_size}"
        f" -> {image_size // patch_size}×{image_size // patch_size}"
    )
    print(f"词表/位置编码尺寸已调整: {resized_keys}")
    print(f"模型参数量: {parameter_count:,}")
    print(f"初始化模型已保存: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

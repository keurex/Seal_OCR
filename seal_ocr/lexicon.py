"""公司名词典约束解码器。

模型仍然是开放字符 OCR；当业务明确知道印章必属于平台注册公司时，可在生成阶段
把候选限制为当前注册公司名单。名单可随新注册公司更新，无需重新训练模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from seal_ocr.data import normalize_company_name


def load_company_names(paths: Sequence[str]) -> List[str]:
    names = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"公司名词典不存在: {path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name = normalize_company_name(line)
            if name:
                names.add(name)
    if not names:
        raise ValueError("公司名词典为空")
    return sorted(names)


class CompanyNameTokenTrie:
    """把公司名字符 token 构造成 ``prefix_allowed_tokens_fn``。"""

    def __init__(
        self,
        names: Iterable[str],
        vocab: Mapping[str, int],
        decoder_start_token_id: int,
        eos_token_id: int,
    ) -> None:
        self.root: Dict[int, dict] = {}
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.eos_token_id = int(eos_token_id)
        self.accepted_names: List[str] = []
        self.rejected_names: List[str] = []

        for name in names:
            token_ids = []
            for character in name:
                token_id = vocab.get(character)
                if token_id is None:
                    token_ids = []
                    break
                token_ids.append(int(token_id))
            if not token_ids:
                self.rejected_names.append(name)
                continue
            node = self.root
            for token_id in [*token_ids, self.eos_token_id]:
                node = node.setdefault(token_id, {})
            self.accepted_names.append(name)

        if not self.accepted_names:
            raise ValueError("词典公司名全部包含模型词表未覆盖字符")

    def allowed_tokens(self, batch_id: int, input_ids) -> List[int]:
        del batch_id
        prefix = [int(token) for token in input_ids.tolist()]
        if prefix and prefix[0] == self.decoder_start_token_id:
            prefix = prefix[1:]
        node = self.root
        for token_id in prefix:
            next_node = node.get(token_id)
            if next_node is None:
                # 正常生成不会走到这里；返回 EOS 可让异常序列安全结束。
                return [self.eos_token_id]
            node = next_node
        allowed = sorted(node)
        return allowed or [self.eos_token_id]

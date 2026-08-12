"""训练标签与 Transformers 损失移位规则的回归测试。"""

from __future__ import annotations

import unittest

try:
    import torch
    from transformers.loss.loss_utils import LOSS_MAPPING
except ModuleNotFoundError:
    torch = None
    LOSS_MAPPING = None

if torch is not None:
    from train import SEQ2SEQ_LOSS_TYPE


@unittest.skipIf(torch is None, "PyTorch 或 Transformers 未安装")
class TrainingLossAlignmentTest(unittest.TestCase):
    def test_seq2seq_loss_keeps_labels_at_the_same_positions(self) -> None:
        """正确 token 已位于同一时间步时，训练损失必须接近零。"""
        vocab_size = 8
        labels = torch.tensor([[0, 4, 5, 2, -100]], dtype=torch.long)
        logits = torch.full(
            (1, labels.shape[1], vocab_size),
            fill_value=-20.0,
            dtype=torch.float32,
        )
        for position, token_id in enumerate(labels[0].tolist()):
            if token_id != -100:
                logits[0, position, token_id] = 20.0

        aligned_loss = LOSS_MAPPING[SEQ2SEQ_LOSS_TYPE](
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
        )
        shifted_loss = LOSS_MAPPING["ForCausalLM"](
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
        )

        self.assertLess(aligned_loss.item(), 1e-6)
        self.assertGreater(shifted_loss.item(), 10.0)


if __name__ == "__main__":
    unittest.main()

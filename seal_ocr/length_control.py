"""印章公司名长度条件。

长度控制分为三个互补部分：

1. Encoder 长度预测头输出离散分布 ``P(L|I)``；
2. 训练时把“剩余字符数”真正加入 decoder 顶层隐藏状态，再重新经过
   输出投影，因此长度条件会影响全部字符 logits，而不只是 EOS；
3. 推理时仅对高置信度长度启用精确 EOS 约束。低置信度样本继续使用
   模型学到的软条件，避免错误长度被硬性放大成乱码。

文字解码器始终只使用模型自己预测的长度分布，并切断这条自条件路径到长度
预测头的梯度。真实长度只监督长度分类损失，避免训练时依赖部署阶段不存在的
oracle 长度，也避免文字损失和长度分类损失同时拉扯长度头。
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """保持前向值不变，只按 ``scale`` 缩放回传到输入的梯度。"""
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("gradient scale 必须在 [0, 1] 范围内")
    detached = tensor.detach()
    return detached + (tensor - detached) * scale


def condition_activation_ratio(global_step: int, activation_steps: int) -> float:
    """预测长度条件的平滑激活比例。

    ``global_step=0`` 时为 0，到 ``activation_steps`` 时升为 1。使用
    smoothstep 而不是线性折线，使激活开始和结束处的变化率都为 0，减少 EOS
    偏置强度变化造成的优化震荡。若激活步数为 0，则从一开始完整启用。
    """
    activation_steps = int(activation_steps)
    if activation_steps <= 0:
        return 1.0
    progress = min(
        1.0,
        max(0.0, float(global_step)) / float(activation_steps),
    )
    return progress * progress * (3.0 - 2.0 * progress)


class LengthConditionModule(nn.Module):
    """长度预测头和剩余字符数条件。

    Args:
        encoder_hidden_size: encoder 输出维度。
        decoder_hidden_size: decoder 隐藏维度。
        max_target_length: 长度分类上限。
        predictor_hidden_size: 长度预测头隐藏维度。
        predictor_dropout: 长度预测头 dropout。
        default_length: 新模块初始化时的长度分布中心。
        eos_bias_init_scale: EOS 初始抑制/鼓励幅度。
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        max_target_length: int = 40,
        predictor_hidden_size: int = 256,
        predictor_dropout: float = 0.1,
        default_length: int = 8,
        eos_bias_init_scale: float = 2.0,
    ):
        super().__init__()
        self.max_target_length = int(max_target_length)
        self.decoder_hidden_size = int(decoder_hidden_size)

        self.length_predictor = nn.Sequential(
            nn.Linear(encoder_hidden_size, predictor_hidden_size),
            nn.GELU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_size, predictor_hidden_size),
            nn.GELU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(predictor_hidden_size, self.max_target_length + 1),
        )
        self._init_weights(default_length)

        # 索引 = remaining + max_target_length，范围 [0, 2*max]。
        self.eos_bias_embedding = nn.Embedding(
            2 * self.max_target_length + 1,
            1,
        )
        self._init_eos_bias(eos_bias_init_scale)

        # 零初始化保证接入已有 OCR 权重时，首次前向的普通字符 logits 不变；
        # 输出投影会让该表从第一步反向传播起获得有效梯度。
        self.hidden_bias_embedding = nn.Embedding(
            2 * self.max_target_length + 1,
            self.decoder_hidden_size,
        )
        nn.init.zeros_(self.hidden_bias_embedding.weight)

    def _init_weights(self, default_length: int) -> None:
        """把初始 softmax 分布真正初始化为指定中心的离散高斯。"""
        last_layer = self.length_predictor[-1]
        nn.init.zeros_(last_layer.weight)
        with torch.no_grad():
            positions = torch.arange(
                self.max_target_length + 1,
                dtype=torch.float32,
            )
            sigma = 3.0
            gaussian_logits = -(
                (positions - float(default_length)) ** 2
            ) / (2.0 * sigma**2)
            last_layer.bias.copy_(gaussian_logits)

    def _init_eos_bias(self, scale: float) -> None:
        """用连续曲线初始化 EOS bias，避免正负两侧全部使用同一常数。"""
        length_limit = self.max_target_length
        with torch.no_grad():
            remaining = (
                torch.arange(2 * length_limit + 1, dtype=torch.float32)
                - length_limit
            )
            # 剩余字符为正时抑制 EOS；越过目标位置后逐步鼓励 EOS。
            bias = -float(scale) * torch.tanh(remaining / 1.5)
            self.eos_bias_embedding.weight.copy_(bias.unsqueeze(-1))

    def forward(self, encoder_hidden_state: torch.Tensor) -> torch.Tensor:
        """从 encoder CLS 表征预测字符数离散分布 logits。"""
        pooled = encoder_hidden_state[:, 0]
        return self.length_predictor(pooled)

    def decode(
        self,
        length_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回离散最大概率长度和对应置信度。"""
        length_probs = F.softmax(length_logits, dim=-1)
        confidence, predicted_length = length_probs.max(dim=-1)
        return predicted_length, confidence

    def expected_length(self, length_logits: torch.Tensor) -> torch.Tensor:
        """返回分布期望，仅用于诊断，不用于精确长度解码。"""
        length_probs = F.softmax(length_logits, dim=-1)
        positions = torch.arange(
            length_logits.size(-1),
            device=length_logits.device,
            dtype=length_probs.dtype,
        )
        return (length_probs * positions).sum(dim=-1)

    def apply_length_condition(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        length_logits: torch.Tensor,
        eos_token_id: int,
        output_projection: nn.Module,
        current_step: int = 0,
        condition_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把剩余长度加入 hidden state，并重新计算全部 token logits。

        文字支路始终使用预测软分布，并对该分布 stop-gradient。这样长度头只由
        明确的长度分类损失训练，不会被字符 CE 反向推向“更容易生成”的错误长度。
        ``condition_scale`` 用于把预测条件从无影响平滑增强到完整影响。

        重新调用 ``output_projection`` 时只把条件产生的 logit 增量加回原始
        ``logits``，避免覆盖 decoder 原有的输出偏置或其它后处理；scale=0 时
        输出严格退化为原始 OCR logits。
        """
        length_limit = self.max_target_length
        _, sequence_length, _ = logits.shape
        device = logits.device

        scale = float(condition_scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError("condition_scale 必须在 [0, 1] 范围内")

        # 关键稳定性约束：OCR loss 不得通过自条件分布更新长度预测头。
        length_probs = F.softmax(length_logits.detach(), dim=-1)

        time_indices = torch.arange(
            current_step,
            current_step + sequence_length,
            device=device,
        )
        length_indices = torch.arange(length_limit + 1, device=device)
        remaining = length_indices.unsqueeze(1) - time_indices.unsqueeze(0)
        remaining_indices = (remaining + length_limit).clamp(
            0,
            2 * length_limit,
        )

        eos_bias_all = self.eos_bias_embedding(remaining_indices).squeeze(-1)
        eos_probabilities = length_probs.to(dtype=eos_bias_all.dtype)
        raw_eos_bias = (
            eos_probabilities.unsqueeze(-1) * eos_bias_all.unsqueeze(0)
        ).sum(dim=1)
        # 限制软 EOS 条件的最大幅度，避免训练后期表值持续放大造成生成震荡。
        eos_bias = 2.0 * torch.tanh(raw_eos_bias / 2.0)

        hidden_bias_all = self.hidden_bias_embedding(remaining_indices)
        hidden_probabilities = length_probs.to(dtype=hidden_bias_all.dtype)
        raw_hidden_bias = (
            hidden_probabilities.unsqueeze(-1).unsqueeze(-1)
            * hidden_bias_all.unsqueeze(0)
        ).sum(dim=1)
        # hidden 条件也做有界残差，避免独立高学习率把 decoder 表征整体推离
        # 原始 TrOCR 分布。
        hidden_bias = torch.tanh(raw_hidden_bias)
        conditioned_hidden_states = hidden_states + scale * hidden_bias.to(
            dtype=hidden_states.dtype,
        )

        base_projected_logits = output_projection(hidden_states)
        conditioned_projected_logits = output_projection(conditioned_hidden_states)
        if conditioned_projected_logits.shape != logits.shape:
            raise RuntimeError(
                "decoder 输出投影形状不匹配："
                f"expected={tuple(logits.shape)}, "
                f"actual={tuple(conditioned_projected_logits.shape)}"
            )
        conditioned_logits = logits + (
            conditioned_projected_logits - base_projected_logits
        ).to(dtype=logits.dtype)
        conditioned_logits = conditioned_logits.clone()
        conditioned_logits[:, :, eos_token_id] = (
            conditioned_logits[:, :, eos_token_id]
            + scale * eos_bias.to(dtype=conditioned_logits.dtype)
        )
        return conditioned_logits, conditioned_hidden_states


class LengthControlLogitsProcessor:
    """对高置信度长度执行精确 EOS 位置约束。

    标签契约为 ``[BOS, c1, ..., cN, EOS]``。处理器按已生成的有效字符
    token 数计数，并忽略 BOS/PAD/UNK 等不会出现在最终文本中的特殊 token，
    因此无论 decoder 是否先重复生成 BOS，都能在 N 个可见字符后结束。
    置信度不足时不做硬约束，forward 内学习到的软长度条件仍然有效。
    """

    def __init__(
        self,
        predicted_length: torch.Tensor,
        confidence: torch.Tensor,
        eos_token_id: int,
        tolerance: int = 0,
        minimum_confidence: float = 0.60,
        ignored_token_ids: Iterable[Optional[int]] = (),
    ):
        if float(tolerance) != int(tolerance):
            raise ValueError("tolerance 必须为整数")
        if int(tolerance) < 0:
            raise ValueError("tolerance 必须大于等于 0")
        if not 0.0 <= float(minimum_confidence) <= 1.0:
            raise ValueError("minimum_confidence 必须在 [0, 1] 范围内")
        if predicted_length.ndim != 1 or confidence.ndim != 1:
            raise ValueError("predicted_length 和 confidence 必须为一维 batch")
        if predicted_length.shape != confidence.shape:
            raise ValueError("predicted_length 与 confidence batch 不一致")
        self.predicted_length = predicted_length.long()
        self.confidence = confidence.float()
        self.eos_token_id = int(eos_token_id)
        self.tolerance = int(tolerance)
        self.minimum_confidence = float(minimum_confidence)
        self.ignored_token_ids = tuple(
            sorted(
                {
                    int(token_id)
                    for token_id in ignored_token_ids
                    if token_id is not None
                }
            )
        )

    @staticmethod
    def _expand_for_generation(
        values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if values.size(0) == batch_size:
            return values
        if values.size(0) <= 0 or batch_size % values.size(0) != 0:
            raise ValueError(
                "长度预测 batch 无法扩展到生成 batch："
                f"length_batch={values.size(0)}, generation_batch={batch_size}"
            )
        repeat_factor = batch_size // values.size(0)
        return values.repeat_interleave(repeat_factor, dim=0)

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = scores.size(0)
        predicted_length = self._expand_for_generation(
            self.predicted_length.to(scores.device),
            batch_size,
        )
        confidence = self._expand_for_generation(
            self.confidence.to(scores.device),
            batch_size,
        )

        active = confidence >= self.minimum_confidence
        if not bool(active.any()):
            return scores

        # 以 decode_text 最终会保留的字符 token 数计数，而不是以生成步数
        # 计数。训练标签显式包含 BOS，首个生成 token 通常仍是 BOS；若按步数
        # 截断，会把这个被忽略的特殊 token 错算成公司名字符。
        content_mask = input_ids.ne(self.eos_token_id)
        for token_id in self.ignored_token_ids:
            content_mask &= input_ids.ne(token_id)
        generated_lengths = content_mask.sum(dim=1)
        block_eos = active & (
            generated_lengths < predicted_length - self.tolerance
        )
        force_eos = active & (
            generated_lengths >= predicted_length + self.tolerance
        )

        scores = scores.clone()
        minimum_score = torch.finfo(scores.dtype).min
        if bool(block_eos.any()):
            scores[block_eos, self.eos_token_id] = minimum_score

        if bool(force_eos.any()):
            forced_scores = torch.full_like(scores, minimum_score)
            # 使用当前最佳局部分数，避免不同 beam 的累计分数被无意义常数覆盖。
            forced_scores[:, self.eos_token_id] = scores.max(dim=-1).values
            scores = torch.where(force_eos.unsqueeze(-1), forced_scores, scores)
        return scores


def compute_true_lengths(
    labels: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """从 ``[BOS, 字符..., EOS, -100...]`` 标签计算字符数。"""
    del bos_token_id, eos_token_id
    valid_count = labels.ne(-100).sum(dim=1).float()
    return (valid_count - 2.0).clamp_min(0.0)


def compute_length_prediction_loss(
    length_logits: torch.Tensor,
    true_lengths: torch.Tensor,
    max_target_length: int,
) -> torch.Tensor:
    """离散长度分类交叉熵。"""
    targets = true_lengths.long().clamp(0, int(max_target_length))
    return F.cross_entropy(length_logits, targets)


def infer_current_step(kwargs: dict) -> int:
    """从 generation cache 推断 decoder 当前时间步。"""
    past_key_values = kwargs.get("past_key_values")
    if past_key_values is None:
        return 0

    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())

    if isinstance(past_key_values, tuple) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, tuple) and first_layer:
            return int(first_layer[0].size(2))
    return 0

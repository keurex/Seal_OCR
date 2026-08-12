"""印章文字、章体及字符阅读顺序的八通道空间辅助监督。

该模块仍只作为共享 encoder 的训练正则，不参与 OCR 解码或部署推理。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seal_ocr.spatial_annotations import SPATIAL_CHANNELS


def _pair(value, name: str) -> Tuple[int, int]:
    """把 int 或二元序列规范为 ``(height, width)``。"""
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, Sequence) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"无法识别 {name}: {value!r}")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} 必须为正数: {pair}")
    return pair


def infer_patch_grid(encoder_config, image_size) -> Tuple[int, int]:
    """根据实际 processor 图片尺寸和 encoder patch 大小推导 patch 网格。"""
    image_height, image_width = _pair(image_size, "image_size")
    patch_height, patch_width = _pair(
        getattr(encoder_config, "patch_size", None),
        "encoder.patch_size",
    )
    if image_height % patch_height or image_width % patch_width:
        raise ValueError(
            "processor 图片尺寸必须能被 encoder patch_size 整除: "
            f"image={(image_height, image_width)}, "
            f"patch={(patch_height, patch_width)}"
        )
    return image_height // patch_height, image_width // patch_width


class SpatialAuxiliaryHead(nn.Module):
    """从 encoder patch token 预测固定八通道标注。

    模块刻意不使用 BatchNorm、running statistics 或 dropout。这样真实后训练冻结
    空间头时，即使父模型处于 train 模式，空间头本身也不会发生隐式状态漂移。
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        head_hidden_size: int,
        grid_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        if encoder_hidden_size <= 0 or head_hidden_size <= 0:
            raise ValueError("空间头隐藏层维度必须大于 0")
        self.grid_size = _pair(grid_size, "grid_size")
        self.output_channels = len(SPATIAL_CHANNELS)
        self.layer_norm = nn.LayerNorm(encoder_hidden_size)
        self.token_projection = nn.Linear(
            encoder_hidden_size,
            head_hidden_size,
        )
        self.refinement = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                head_hidden_size,
                head_hidden_size,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                head_hidden_size,
                self.output_channels,
                kernel_size=1,
            ),
        )
        output_layer = self.refinement[-1]
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.001)
        nn.init.constant_(output_layer.bias, -2.0)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states 必须为 [batch, tokens, hidden]，"
                f"实际为 {tuple(encoder_hidden_states.shape)}"
            )
        grid_height, grid_width = self.grid_size
        patch_count = grid_height * grid_width
        sequence_length = encoder_hidden_states.shape[1]
        prefix_count = sequence_length - patch_count
        # DeiT 通常包含 CLS 和 distillation 两个前缀 token。允许少量其它前缀，
        # 但拒绝静默截断大量 token，避免模型尺寸/processor 配置错误。
        if prefix_count < 0 or prefix_count > 8:
            raise ValueError(
                "encoder token 数与空间网格不匹配: "
                f"tokens={sequence_length}, grid={self.grid_size}, "
                f"prefix={prefix_count}"
            )
        patch_tokens = encoder_hidden_states[:, prefix_count:, :]
        projected = self.token_projection(self.layer_norm(patch_tokens))
        batch_size, _, hidden_size = projected.shape
        feature_map = projected.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            grid_height,
            grid_width,
        )
        return self.refinement(feature_map)


def _weighted_mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回正样本有界加权 BCE 和 soft Dice loss。"""
    targets = targets.float()
    positive_fraction = targets.mean(dim=(-2, -1), keepdim=True)
    positive_weight = (
        (1.0 - positive_fraction) / positive_fraction.clamp_min(1e-4)
    ).clamp(min=1.0, max=8.0).detach()
    element_weight = torch.where(
        targets > 0.5,
        positive_weight,
        torch.ones_like(positive_weight),
    )
    bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        reduction="none",
    )
    bce = (bce * element_weight).sum() / element_weight.sum().clamp_min(1.0)

    probabilities = torch.sigmoid(logits.float())
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(
        dim=(-2, -1)
    )
    dice_loss = 1.0 - (
        (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    ).mean()
    return bce, dice_loss


def _heatmap_weights(
    targets: torch.Tensor,
    maximum_foreground_boost: float,
) -> torch.Tensor:
    """按每张图目标密度自适应提高稀疏 heatmap 峰值权重。"""
    target_density = targets.mean(dim=(-2, -1), keepdim=True)
    foreground_boost = (
        (1.0 - target_density) / target_density.clamp_min(1e-4)
    ).clamp(min=4.0, max=float(maximum_foreground_boost)).detach()
    return 1.0 + foreground_boost * targets


def _heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    maximum_foreground_boost: float = 4.0,
) -> torch.Tensor:
    """对 Gaussian heatmap 做前景加权的平滑回归。"""
    targets = targets.float().clamp(0.0, 1.0)
    probabilities = torch.sigmoid(logits.float())
    # 高响应位置更重要，但保持所有背景位置有梯度，抑制整图虚假高响应。
    weights = _heatmap_weights(targets, maximum_foreground_boost)
    element_loss = F.smooth_l1_loss(
        probabilities,
        targets,
        reduction="none",
        beta=0.1,
    )
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def _weighted_heatmap_mae(
    logits: torch.Tensor,
    targets: torch.Tensor,
    maximum_foreground_boost: float = 4.0,
) -> torch.Tensor:
    """返回与训练目标同权重的 MAE，避免大面积零背景掩盖前景错误。"""
    targets = targets.float().clamp(0.0, 1.0)
    probabilities = torch.sigmoid(logits.float())
    weights = _heatmap_weights(targets, maximum_foreground_boost)
    return (
        (probabilities - targets).abs() * weights
    ).sum() / weights.sum().clamp_min(1.0)


def _masked_progress_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """只在确有公司文字的 patch 回归字符阅读进度。"""
    targets = targets.float().clamp(0.0, 1.0)
    probabilities = torch.sigmoid(logits.float())
    foreground = (targets > 0).float()
    denominator = foreground.sum().clamp_min(1.0)
    loss = F.smooth_l1_loss(
        probabilities,
        targets,
        reduction="none",
        beta=0.1,
    )
    loss = (loss * foreground).sum() / denominator
    mae = ((probabilities - targets).abs() * foreground).sum() / denominator
    return loss, mae


def _heatmap_mass_relative_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """比较预测/目标响应总量，作为字符中心数量与虚假背景响应的代理指标。"""
    predicted_mass = torch.sigmoid(logits.float()).sum(dim=(-2, -1))
    target_mass = targets.float().sum(dim=(-2, -1))
    valid = target_mass > 1e-6
    if not bool(valid.any()):
        return logits.new_zeros(())
    relative_error = (
        (predicted_mass - target_mass).abs()
        / target_mass.clamp_min(1e-6)
    )
    return relative_error[valid].mean()


def _normalized_argmax_distance(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算预测与目标峰值间、按 patch 网格对角线归一化的距离。"""
    if logits.ndim != 3:
        raise ValueError("首尾位置距离输入必须为 [batch, height, width]")
    batch_size, height, width = logits.shape
    target_flat = targets.reshape(batch_size, -1)
    valid = target_flat.amax(dim=1) > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    predicted_indices = logits.reshape(batch_size, -1).argmax(dim=1)
    target_indices = target_flat.argmax(dim=1)
    predicted_y = torch.div(predicted_indices, width, rounding_mode="floor")
    predicted_x = predicted_indices % width
    target_y = torch.div(target_indices, width, rounding_mode="floor")
    target_x = target_indices % width
    distances = torch.sqrt(
        (predicted_y.float() - target_y.float()).square()
        + (predicted_x.float() - target_x.float()).square()
    )
    diagonal = max(((height - 1) ** 2 + (width - 1) ** 2) ** 0.5, 1.0)
    return (distances[valid] / diagonal).mean()


def _binary_overlap_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    predicted = torch.sigmoid(logits.float()) > 0.5
    expected = targets > 0.5
    intersection = (predicted & expected).sum(dim=(-2, -1)).float()
    predicted_area = predicted.sum(dim=(-2, -1)).float()
    expected_area = expected.sum(dim=(-2, -1)).float()
    union = (predicted | expected).sum(dim=(-2, -1)).float()
    dice = (2.0 * intersection + 1e-5) / (
        predicted_area + expected_area + 1e-5
    )
    iou = (intersection + 1e-5) / (union + 1e-5)
    return dice.mean(), iou.mean()


def compute_spatial_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    text_weight: float = 1.0,
    stamp_weight: float = 0.5,
    heatmap_weight: float = 0.25,
    character_weight: float = 1.0,
    return_metrics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算空间辅助损失；评估时可同时返回诊断指标。"""
    if logits.shape != targets.shape:
        raise ValueError(
            "空间头输出与 target 形状不一致: "
            f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
        )
    if logits.ndim != 4 or logits.shape[1] != len(SPATIAL_CHANNELS):
        raise ValueError(
            "空间监督必须为 [batch, 8, grid_h, grid_w]，"
            f"实际为 {tuple(logits.shape)}"
        )
    if text_weight <= 0 or stamp_weight <= 0:
        raise ValueError("text_weight 和 stamp_weight 必须大于 0")
    if heatmap_weight < 0:
        raise ValueError("heatmap_weight 必须大于等于 0")
    if character_weight <= 0:
        raise ValueError("character_weight 必须大于 0")

    text_bce, text_dice_loss = _weighted_mask_loss(
        logits[:, 0], targets[:, 0]
    )
    stamp_bce, stamp_dice_loss = _weighted_mask_loss(
        logits[:, 1], targets[:, 1]
    )
    text_heatmap_loss = _heatmap_loss(logits[:, 2], targets[:, 2])
    stamp_heatmap_loss = _heatmap_loss(logits[:, 3], targets[:, 3])
    text_loss = text_bce + text_dice_loss + heatmap_weight * text_heatmap_loss
    stamp_loss = stamp_bce + stamp_dice_loss + heatmap_weight * stamp_heatmap_loss
    character_center_loss = _heatmap_loss(
        logits[:, 4],
        targets[:, 4],
        maximum_foreground_boost=32.0,
    )
    reading_progress_loss, reading_progress_mae = _masked_progress_loss(
        logits[:, 5], targets[:, 5]
    )
    first_character_loss = _heatmap_loss(
        logits[:, 6],
        targets[:, 6],
        maximum_foreground_boost=64.0,
    )
    last_character_loss = _heatmap_loss(
        logits[:, 7],
        targets[:, 7],
        maximum_foreground_boost=64.0,
    )
    character_loss = (
        character_center_loss
        + reading_progress_loss
        + 0.5 * first_character_loss
        + 0.5 * last_character_loss
    ) / 3.0
    total_loss = (
        text_weight * text_loss
        + stamp_weight * stamp_loss
        + character_weight * character_loss
    ) / (text_weight + stamp_weight + character_weight)

    if not return_metrics:
        return total_loss, {}

    with torch.no_grad():
        character_metrics = {
            "spatial_character_loss": character_loss.detach(),
            "spatial_character_center_heatmap_loss": (
                character_center_loss.detach()
            ),
            "spatial_character_reading_progress_loss": (
                reading_progress_loss.detach()
            ),
            "spatial_first_character_heatmap_loss": (
                first_character_loss.detach()
            ),
            "spatial_last_character_heatmap_loss": (
                last_character_loss.detach()
            ),
            "spatial_character_center_heatmap_mae": _weighted_heatmap_mae(
                logits[:, 4], targets[:, 4], maximum_foreground_boost=32.0,
            ),
            "spatial_character_center_mass_relative_error": (
                _heatmap_mass_relative_error(logits[:, 4], targets[:, 4])
            ),
            "spatial_character_reading_progress_mae": reading_progress_mae.detach(),
            "spatial_first_character_heatmap_mae": _weighted_heatmap_mae(
                logits[:, 6], targets[:, 6], maximum_foreground_boost=64.0,
            ),
            "spatial_last_character_heatmap_mae": _weighted_heatmap_mae(
                logits[:, 7], targets[:, 7], maximum_foreground_boost=64.0,
            ),
            "spatial_first_character_distance": _normalized_argmax_distance(
                logits[:, 6], targets[:, 6],
            ),
            "spatial_last_character_distance": _normalized_argmax_distance(
                logits[:, 7], targets[:, 7],
            ),
        }
        text_dice, text_iou = _binary_overlap_metrics(
            logits[:, 0], targets[:, 0]
        )
        stamp_dice, stamp_iou = _binary_overlap_metrics(
            logits[:, 1], targets[:, 1]
        )
        metrics = {
            "spatial_loss": total_loss.detach(),
            "spatial_text_mask_bce": text_bce.detach(),
            "spatial_text_mask_dice_loss": text_dice_loss.detach(),
            "spatial_stamp_mask_bce": stamp_bce.detach(),
            "spatial_stamp_mask_dice_loss": stamp_dice_loss.detach(),
            "spatial_text_heatmap_loss": text_heatmap_loss.detach(),
            "spatial_stamp_heatmap_loss": stamp_heatmap_loss.detach(),
            "spatial_text_mask_dice": text_dice,
            "spatial_text_mask_iou": text_iou,
            "spatial_stamp_mask_dice": stamp_dice,
            "spatial_stamp_mask_iou": stamp_iou,
            "spatial_text_heatmap_mae": _weighted_heatmap_mae(
                logits[:, 2], targets[:, 2]
            ),
            "spatial_stamp_heatmap_mae": _weighted_heatmap_mae(
                logits[:, 3], targets[:, 3]
            ),
            **character_metrics,
        }
    return total_loss, metrics

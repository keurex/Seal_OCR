"""正确长度条件与高置信精确 EOS 约束的回归测试。"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from seal_ocr.length_control import (
        LengthConditionModule,
        LengthControlLogitsProcessor,
        compute_length_prediction_loss,
        condition_activation_ratio,
        scale_gradient,
    )


@unittest.skipIf(torch is None, "PyTorch 未安装")
class LengthControlTest(unittest.TestCase):
    def test_default_distribution_is_centered_on_requested_length(self) -> None:
        module = LengthConditionModule(
            encoder_hidden_size=4,
            decoder_hidden_size=3,
            max_target_length=20,
            predictor_hidden_size=6,
            predictor_dropout=0.0,
            default_length=8,
        )
        logits = module(torch.zeros(2, 5, 4))
        predicted_length, _ = module.decode(logits)
        self.assertEqual(predicted_length.tolist(), [8, 8])

    def test_predicted_condition_activation_uses_smoothstep(self) -> None:
        self.assertEqual(condition_activation_ratio(0, 100), 0.0)
        self.assertAlmostEqual(
            condition_activation_ratio(25, 100),
            0.15625,
        )
        self.assertEqual(condition_activation_ratio(50, 100), 0.5)
        self.assertEqual(condition_activation_ratio(100, 100), 1.0)
        self.assertEqual(condition_activation_ratio(200, 100), 1.0)
        self.assertEqual(condition_activation_ratio(0, 0), 1.0)

    def test_gradient_scaling_preserves_forward_value(self) -> None:
        value = torch.tensor([2.0], requires_grad=True)
        scaled = scale_gradient(value, 0.1)
        self.assertEqual(scaled.item(), value.item())
        scaled.sum().backward()
        self.assertAlmostEqual(value.grad.item(), 0.1, places=6)

    def test_hidden_length_condition_receives_text_loss_gradient(self) -> None:
        torch.manual_seed(7)
        module = LengthConditionModule(
            encoder_hidden_size=4,
            decoder_hidden_size=3,
            max_target_length=5,
            predictor_hidden_size=6,
            predictor_dropout=0.0,
            default_length=2,
        )
        projection = torch.nn.Linear(3, 9, bias=False)
        hidden_states = torch.randn(1, 4, 3)
        logits = projection(hidden_states)
        length_logits = module(torch.randn(1, 6, 4))
        conditioned_logits, _ = module.apply_length_condition(
            logits=logits,
            hidden_states=hidden_states,
            length_logits=length_logits,
            eos_token_id=2,
            output_projection=projection,
            current_step=0,
            condition_scale=1.0,
        )
        targets = torch.tensor([[0, 5, 6, 2]])
        torch.nn.functional.cross_entropy(
            conditioned_logits.reshape(-1, 9),
            targets.reshape(-1),
        ).backward()
        gradient = module.hidden_bias_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        predictor_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in module.length_predictor.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(predictor_gradient, 0.0)

    def test_length_classification_loss_still_updates_predictor(self) -> None:
        module = LengthConditionModule(
            encoder_hidden_size=4,
            decoder_hidden_size=3,
            max_target_length=5,
            predictor_hidden_size=6,
            predictor_dropout=0.0,
            default_length=2,
        )
        length_logits = module(torch.randn(2, 6, 4))
        compute_length_prediction_loss(
            length_logits,
            torch.tensor([2.0, 4.0]),
            module.max_target_length,
        ).backward()
        gradient = module.length_predictor[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_zero_condition_scale_preserves_original_logits(self) -> None:
        torch.manual_seed(11)
        module = LengthConditionModule(
            encoder_hidden_size=4,
            decoder_hidden_size=3,
            max_target_length=5,
            predictor_hidden_size=6,
            predictor_dropout=0.0,
            default_length=2,
        )
        with torch.no_grad():
            module.hidden_bias_embedding.weight.normal_()
            module.eos_bias_embedding.weight.normal_()
        projection = torch.nn.Linear(3, 9, bias=False)
        hidden_states = torch.randn(2, 4, 3)
        logits = projection(hidden_states).detach()
        length_logits = module(torch.randn(2, 6, 4))
        conditioned_logits, conditioned_hidden = module.apply_length_condition(
            logits=logits,
            hidden_states=hidden_states,
            length_logits=length_logits,
            eos_token_id=2,
            output_projection=projection,
            current_step=0,
            condition_scale=0.0,
        )
        self.assertTrue(torch.equal(conditioned_logits, logits))
        self.assertTrue(torch.equal(conditioned_hidden, hidden_states))

    def test_exact_constraint_counts_visible_tokens_not_bos(self) -> None:
        processor = LengthControlLogitsProcessor(
            predicted_length=torch.tensor([2]),
            confidence=torch.tensor([0.9]),
            eos_token_id=2,
            tolerance=0,
            minimum_confidence=0.6,
            ignored_token_ids=(0, 1, 3),
        )
        scores = torch.zeros(1, 8)
        before_text = processor(torch.tensor([[0, 0]]), scores)
        after_one_character = processor(torch.tensor([[0, 0, 5]]), scores)
        after_two_characters = processor(
            torch.tensor([[0, 0, 5, 6]]), scores
        )
        minimum_score = torch.finfo(scores.dtype).min
        self.assertEqual(before_text[0, 2].item(), minimum_score)
        self.assertEqual(after_one_character[0, 2].item(), minimum_score)
        self.assertGreater(after_two_characters[0, 2].item(), minimum_score)
        self.assertTrue(
            torch.all(after_two_characters[0, [0, 1, 3, 4, 5, 6, 7]] == minimum_score)
        )

    def test_low_confidence_and_beam_expansion_are_safe(self) -> None:
        scores = torch.randn(4, 8)
        low_confidence = LengthControlLogitsProcessor(
            predicted_length=torch.tensor([2]),
            confidence=torch.tensor([0.2]),
            eos_token_id=2,
            ignored_token_ids=(0, 1, 3),
        )
        unchanged = low_confidence(torch.tensor([[0], [0], [0], [0]]), scores)
        self.assertTrue(torch.equal(unchanged, scores))

        expanded = LengthControlLogitsProcessor(
            predicted_length=torch.tensor([2, 3]),
            confidence=torch.tensor([0.9, 0.9]),
            eos_token_id=2,
            ignored_token_ids=(0, 1, 3),
        )
        processed = expanded(
            torch.tensor(
                [
                    [0, 0, 5, 6],
                    [0, 0, 5, 6],
                    [0, 0, 5, 6],
                    [0, 0, 5, 6],
                ]
            ),
            scores,
        )
        minimum_score = torch.finfo(scores.dtype).min
        self.assertTrue(torch.all(processed[:2, 2] > minimum_score))
        self.assertTrue(torch.all(processed[2:, 2] == minimum_score))


if __name__ == "__main__":
    unittest.main()

import pytest
import torch

from src.rl.grpo_trainer import (
    AgenticGRPOTrainer,
    GRPOConfig,
    compute_grpo_loss,
    masked_mean,
    normalize_group_rewards,
    rewards_to_tensor,
)


def test_group_rewards_are_normalized_per_prompt_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 14.0])

    advantages = normalize_group_rewards(rewards, group_size=3)

    expected_first = torch.tensor([-1.2247448, 0.0, 1.2247448])
    expected_second = torch.tensor([-0.70710677, -0.70710677, 1.4142135])
    assert torch.allclose(advantages[:3], expected_first, atol=1e-6)
    assert torch.allclose(advantages[3:], expected_second, atol=1e-6)


def test_group_reward_normalization_handles_constant_rewards():
    rewards = torch.tensor([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

    advantages = normalize_group_rewards(rewards, group_size=3)

    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_group_reward_normalization_requires_complete_groups():
    with pytest.raises(ValueError, match="divisible"):
        normalize_group_rewards(torch.tensor([1.0, 2.0, 3.0]), group_size=2)


def test_masked_mean_respects_mask():
    values = torch.tensor([[1.0, 100.0], [3.0, 100.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    assert masked_mean(values, mask) == torch.tensor(2.0)


def test_grpo_loss_respects_loss_mask():
    logprobs_new = torch.log(torch.tensor([[0.8, 0.1]], dtype=torch.float32))
    logprobs_old = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    advantages = torch.tensor([1.0])
    loss_mask = torch.tensor([[1.0, 0.0]])

    out = compute_grpo_loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_eps=0.2,
    )

    assert out.masked_token_count == torch.tensor(1.0)
    assert out.policy_loss == pytest.approx(-1.2)
    assert out.loss == pytest.approx(-1.2)


def test_grpo_loss_uses_clipped_objective_for_negative_advantage():
    logprobs_new = torch.log(torch.tensor([[0.1]], dtype=torch.float32))
    logprobs_old = torch.log(torch.tensor([[0.5]], dtype=torch.float32))
    advantages = torch.tensor([-1.0])
    loss_mask = torch.tensor([[1.0]])

    out = compute_grpo_loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_eps=0.2,
    )

    assert out.policy_loss == pytest.approx(0.8)
    assert out.clip_fraction == pytest.approx(1.0)


def test_grpo_loss_can_backward():
    logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    logprobs_new = torch.log_softmax(logits, dim=-1)
    logprobs_old = torch.zeros_like(logprobs_new).detach()
    advantages = torch.tensor([1.0])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

    out = compute_grpo_loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=advantages,
        loss_mask=loss_mask,
    )
    out.loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_grpo_loss_adds_optional_kl_penalty():
    logprobs_new = torch.tensor([[-0.2, -0.4]])
    logprobs_old = torch.tensor([[-0.2, -0.4]])
    ref_logprobs = torch.tensor([[-0.3, -0.7]])
    advantages = torch.tensor([0.0])
    loss_mask = torch.tensor([[1.0, 1.0]])

    out = compute_grpo_loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=advantages,
        loss_mask=loss_mask,
        ref_logprobs=ref_logprobs,
        beta_kl=0.5,
    )

    assert out.policy_loss == pytest.approx(0.0)
    assert out.kl_loss == pytest.approx(0.2)
    assert out.loss == pytest.approx(0.1)


def test_trainer_has_no_value_model_requirement():
    trainer = AgenticGRPOTrainer(GRPOConfig(num_generations=2))

    assert not hasattr(trainer, "value_model")
    rewards = rewards_to_tensor([1.0, 3.0])
    advantages = trainer.normalize_rewards(rewards)

    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0]))


def test_trainer_loss_delegates_to_compute_grpo_loss():
    trainer = AgenticGRPOTrainer(GRPOConfig(clip_eps=0.1))
    logprobs_new = torch.log(torch.tensor([[0.7]], dtype=torch.float32))
    logprobs_old = torch.log(torch.tensor([[0.5]], dtype=torch.float32))

    out = trainer.loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=torch.tensor([1.0]),
        loss_mask=torch.tensor([[1.0]]),
    )

    assert out.policy_loss == pytest.approx(-1.1)

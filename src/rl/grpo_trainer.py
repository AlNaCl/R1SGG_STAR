"""GRPO loss utilities and lightweight trainer skeleton for Agentic RLVR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for group-relative policy optimization."""

    num_generations: int = 4
    clip_eps: float = 0.2
    beta_kl: float = 0.0
    advantage_eps: float = 1e-8


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    masked_token_count: torch.Tensor


@dataclass
class AgenticGRPOTrainer:
    """Small orchestration shell for future model-backed Agentic GRPO training.

    This class intentionally does not own a value model. It expects rollout and
    logprob functions to be injected by later phases, keeping Phase 5 testable
    without loading a policy model.
    """

    config: GRPOConfig = field(default_factory=GRPOConfig)
    rollout_fn: Callable[..., Any] | None = None
    logprob_fn: Callable[..., torch.Tensor] | None = None
    ref_logprob_fn: Callable[..., torch.Tensor] | None = None

    def normalize_rewards(self, rewards: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
        return normalize_group_rewards(rewards, group_size or self.config.num_generations, self.config.advantage_eps)

    def loss(
        self,
        logprobs_new: torch.Tensor,
        logprobs_old: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        ref_logprobs: torch.Tensor | None = None,
    ) -> GRPOLossOutput:
        return compute_grpo_loss(
            logprobs_new=logprobs_new,
            logprobs_old=logprobs_old,
            advantages=advantages,
            loss_mask=loss_mask,
            ref_logprobs=ref_logprobs,
            clip_eps=self.config.clip_eps,
            beta_kl=self.config.beta_kl,
        )


def normalize_group_rewards(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards independently inside each prompt group."""

    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if rewards.numel() % group_size != 0:
        raise ValueError("number of rewards must be divisible by group_size")
    original_shape = rewards.shape
    grouped = rewards.float().reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(original_shape)


def expand_group_advantages(advantages: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sequence advantages to token logprob shape."""

    if advantages.ndim == target.ndim:
        return advantages.to(device=target.device, dtype=target.dtype)
    if advantages.ndim != 1 or target.ndim < 2:
        raise ValueError("advantages must be 1D or already match target dimensions")
    if advantages.shape[0] != target.shape[0]:
        raise ValueError("advantages length must match target batch dimension")
    view_shape = (advantages.shape[0],) + (1,) * (target.ndim - 1)
    return advantages.to(device=target.device, dtype=target.dtype).reshape(view_shape)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over entries where mask is non-zero."""

    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom


def compute_token_kl(logprobs_new: torch.Tensor, ref_logprobs: torch.Tensor | None) -> torch.Tensor:
    """Approximate per-token KL from sampled-token log probabilities."""

    if ref_logprobs is None:
        return torch.zeros_like(logprobs_new)
    if ref_logprobs.shape != logprobs_new.shape:
        raise ValueError("ref_logprobs must match logprobs_new shape")
    return logprobs_new - ref_logprobs.to(device=logprobs_new.device, dtype=logprobs_new.dtype)


def compute_grpo_loss(
    *,
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    beta_kl: float = 0.0,
) -> GRPOLossOutput:
    """Compute clipped GRPO policy loss with token-level masking."""

    if logprobs_new.shape != logprobs_old.shape:
        raise ValueError("logprobs_new and logprobs_old must have the same shape")
    if loss_mask.shape != logprobs_new.shape:
        raise ValueError("loss_mask must match logprob shape")
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative")
    if beta_kl < 0:
        raise ValueError("beta_kl must be non-negative")

    adv = expand_group_advantages(advantages, logprobs_new)
    logprobs_old = logprobs_old.to(device=logprobs_new.device, dtype=logprobs_new.dtype)
    ratio = torch.exp(logprobs_new - logprobs_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped = ratio * adv
    clipped = clipped_ratio * adv
    policy_per_token = -torch.minimum(unclipped, clipped)
    policy_loss = masked_mean(policy_per_token, loss_mask)

    token_kl = compute_token_kl(logprobs_new, ref_logprobs)
    kl_loss = masked_mean(token_kl, loss_mask) if ref_logprobs is not None else torch.zeros((), device=logprobs_new.device, dtype=logprobs_new.dtype)
    loss = policy_loss + float(beta_kl) * kl_loss
    mask = loss_mask.to(device=logprobs_new.device, dtype=logprobs_new.dtype)
    clip_fraction = masked_mean((torch.abs(ratio - 1.0) > clip_eps).to(logprobs_new.dtype), mask)
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        approx_kl=kl_loss.detach(),
        clip_fraction=clip_fraction.detach(),
        masked_token_count=mask.sum().detach(),
    )


def rewards_to_tensor(rewards: Iterable[float], device: torch.device | str | None = None) -> torch.Tensor:
    """Convert scalar rewards to a float tensor for GRPO utilities."""

    return torch.tensor(list(rewards), dtype=torch.float32, device=device)



def _resolve_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    import os
    import re

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(name, default)

    return pattern.sub(repl, value)


def _resolve_env_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_config(v) for v in value]
    return _resolve_env_value(value)


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return _resolve_env_config(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})


def _toy_generator_factory(outputs: list[str]) -> Callable[[list[dict[str, Any]], dict[str, Any], int], str]:
    def generate(history: list[dict[str, Any]], sample: dict[str, Any], step: int) -> str:
        return outputs[min(step, len(outputs) - 1)]

    return generate


def run_dry_run(config_path: str | None = None, output_root: str | None = None) -> dict[str, Any]:
    """Run a tiny rollout + reward + GRPO loss + backward smoke test."""

    from pathlib import Path
    import json
    import os

    from PIL import Image

    from src.rl.agentic_rollout import RolloutConfig, run_agentic_rollout
    from src.rl.mask_utils import build_loss_mask
    from src.rl.paths import ensure_output_dirs, resolve_agentic_paths

    raw_config = _load_yaml_config(config_path)
    grpo_cfg_raw = raw_config.get("grpo", {}) if isinstance(raw_config, dict) else {}
    reward_cfg = raw_config.get("reward", {}) if isinstance(raw_config, dict) else {}
    grpo_cfg = GRPOConfig(
        num_generations=2,
        clip_eps=float(grpo_cfg_raw.get("clip_eps", 0.2)),
        beta_kl=float(grpo_cfg_raw.get("beta_kl", 0.0)),
        advantage_eps=float(grpo_cfg_raw.get("advantage_eps", 1e-8)),
    )

    if output_root:
        os.environ["OUTPUT_ROOT"] = output_root
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)

    image = Image.new("RGB", (64, 64), "white")
    samples = [
        {
            "id": "toy_0",
            "prompt": "Answer yes or no after inspecting the image.",
            "image": image,
            "width": 64,
            "height": 64,
            "task_type": "true_false",
            "answer": "yes",
        },
        {
            "id": "toy_1",
            "prompt": "Identify whether the object is present.",
            "image": image,
            "width": 64,
            "height": 64,
            "task_type": "multiple_choice",
            "answer": "b",
        },
    ]
    generation_groups = [
        [
            [
                '{"thought": "inspect center", "action": "zoom_in", "bbox": [8, 8, 40, 40]}',
                '{"thought": "evidence found", "action": "final_answer", "answer": "yes"}',
            ],
            ['{"thought": "answer directly", "action": "final_answer", "answer": "no"}'],
        ],
        [
            [
                '{"thought": "inspect", "action": "zoom_in", "bbox": [0.1, 0.1, 0.7, 0.7]}',
                '{"thought": "choose option", "action": "final_answer", "answer": "b"}',
            ],
            ['{"thought": "guess", "action": "final_answer", "answer": "a"}'],
        ],
    ]

    rollout_results = []
    rewards = []
    masks = []
    max_len = 0
    for sample, group in zip(samples, generation_groups):
        for output_sequence in group:
            rollout_cfg = RolloutConfig(
                max_tool_steps=1,
                coord_type="normalized" if sample["id"] == "toy_1" and len(output_sequence) > 1 else "pixel",
                reward_config=reward_cfg,
            )
            result = run_agentic_rollout(sample, _toy_generator_factory(output_sequence), rollout_cfg)
            rollout_results.append(result)
            rewards.append(float(result.reward["reward"]))
            max_len = max(max_len, result.token_spans[-1].end)

    for result in rollout_results:
        seq_len = result.token_spans[-1].end
        input_ids = torch.arange(seq_len)
        mask = build_loss_mask(input_ids, result.token_spans)
        masks.append(torch.nn.functional.pad(mask, (0, max_len - seq_len)))

    rewards_t = rewards_to_tensor(rewards)
    advantages = normalize_group_rewards(rewards_t, group_size=grpo_cfg.num_generations, eps=grpo_cfg.advantage_eps)
    loss_mask = torch.stack(masks, dim=0)

    logits = torch.nn.Parameter(torch.zeros_like(loss_mask))
    logprobs_new = torch.log_softmax(logits, dim=-1)
    logprobs_old = torch.zeros_like(logprobs_new).detach()
    loss_out = compute_grpo_loss(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_eps=grpo_cfg.clip_eps,
        beta_kl=grpo_cfg.beta_kl,
    )
    loss_out.loss.backward()

    summary = {
        "dry_run": True,
        "num_samples": len(samples),
        "num_generations": grpo_cfg.num_generations,
        "num_trajectories": len(rollout_results),
        "rewards": rewards,
        "advantages": [float(x) for x in advantages.tolist()],
        "mean_reward": float(rewards_t.mean().item()),
        "loss": float(loss_out.loss.detach().item()),
        "policy_loss": float(loss_out.policy_loss.detach().item()),
        "kl_loss": float(loss_out.kl_loss.detach().item()),
        "clip_fraction": float(loss_out.clip_fraction.detach().item()),
        "masked_token_count": float(loss_out.masked_token_count.detach().item()),
        "grad_norm": float(logits.grad.norm().item()) if logits.grad is not None else 0.0,
        "valid_json_rate": sum(1 for r in rollout_results if r.reward["is_valid_json"]) / len(rollout_results),
        "zoom_in_usage_rate": sum(1 for r in rollout_results if r.trajectory["used_zoom"]) / len(rollout_results),
        "invalid_bbox_rate": sum(1 for r in rollout_results if r.reward["invalid_bbox"]) / len(rollout_results),
        "mean_trajectory_length": sum(r.trajectory["num_steps"] for r in rollout_results) / len(rollout_results),
    }
    log_path = paths.output_root / "logs" / "dry_run_agentic_grpo.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["log_path"] = str(log_path)
    return summary




def _config_section(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key, {}) if isinstance(raw_config, dict) else {}
    return value if isinstance(value, dict) else {}


def run_train_smoke(config_path: str | None = None, output_root: str | None = None) -> dict[str, Any]:
    """Run a tiny optimizer smoke test over real RLVR dataset metadata.

    This verifies the real dataset adapter, GRPO loss, backward, optimizer step,
    log writing, and checkpoint writing without loading a large policy model.
    """

    import json
    import os
    from datetime import datetime

    from src.data.rlvr_dataset import RLVRDatasetConfig, load_rlvr_dataset, summarize_rlvr_dataset
    from src.rl.paths import ensure_output_dirs, resolve_agentic_paths

    raw_config = _load_yaml_config(config_path)
    grpo_raw = _config_section(raw_config, "grpo")
    dataset_raw = _config_section(raw_config, "rlvr_dataset")
    reward_raw = _config_section(raw_config, "reward")
    if output_root:
        os.environ["OUTPUT_ROOT"] = output_root
    paths = resolve_agentic_paths(create_output=True)
    ensure_output_dirs(paths)

    split = str(dataset_raw.get("train_split", dataset_raw.get("split", "train")))
    dataset_cfg = RLVRDatasetConfig(
        source=str(dataset_raw.get("source", "hf_closed")),
        dataset_path=dataset_raw.get("dataset_path") or str(paths.dataset_path),
        jsonl_dir=dataset_raw.get("jsonl_dir") or str(paths.jsonl_closed_dir),
        split=split,
        task_type=str(dataset_raw.get("task_type", "scene_graph")),
        prompt_field=str(dataset_raw.get("prompt_field", "prompt_close")),
        require_image_exists=bool(dataset_raw.get("require_image_exists", True)),
    )
    dataset = load_rlvr_dataset(dataset_cfg, paths=paths)
    if len(dataset) < 1:
        raise ValueError("RLVR dataset is empty; cannot run train smoke")

    num_samples = int(grpo_raw.get("smoke_num_samples", 2))
    num_generations = int(grpo_raw.get("smoke_num_generations", 2))
    max_steps = int(grpo_raw.get("smoke_train_steps", 1))
    seq_len = int(grpo_raw.get("smoke_seq_len", 8))
    learning_rate = float(grpo_raw.get("learning_rate", 1e-6))
    grpo_cfg = GRPOConfig(
        num_generations=num_generations,
        clip_eps=float(grpo_raw.get("clip_eps", 0.2)),
        beta_kl=float(grpo_raw.get("beta_kl", 0.0)),
        advantage_eps=float(grpo_raw.get("advantage_eps", grpo_raw.get("eps", 1e-8))),
    )

    selected = [dataset[i % len(dataset)] for i in range(num_samples)]
    rewards = []
    for sample in selected:
        rel_count = len(sample.get("relationships") or [])
        obj_count = len(sample.get("objects") or [])
        base = 1.0 if rel_count >= 0 and obj_count >= 0 else 0.0
        for generation_idx in range(num_generations):
            rewards.append(base if generation_idx == 0 else float(reward_raw.get("lambda_format", 0.1)))
    rewards_t = rewards_to_tensor(rewards)
    advantages = normalize_group_rewards(rewards_t, group_size=num_generations, eps=grpo_cfg.advantage_eps)

    parameter = torch.nn.Parameter(torch.zeros((len(rewards), seq_len), dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    loss_history = []
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logprobs_new = parameter
        logprobs_old = torch.zeros_like(logprobs_new).detach()
        loss_mask = torch.ones_like(logprobs_new)
        loss_out = compute_grpo_loss(
            logprobs_new=logprobs_new,
            logprobs_old=logprobs_old,
            advantages=advantages,
            loss_mask=loss_mask,
            clip_eps=grpo_cfg.clip_eps,
            beta_kl=grpo_cfg.beta_kl,
        )
        loss_out.loss.backward()
        grad_norm = float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0
        optimizer.step()
        loss_history.append(
            {
                "loss": float(loss_out.loss.detach().item()),
                "policy_loss": float(loss_out.policy_loss.detach().item()),
                "kl_loss": float(loss_out.kl_loss.detach().item()),
                "clip_fraction": float(loss_out.clip_fraction.detach().item()),
                "grad_norm": grad_norm,
            }
        )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    ckpt_dir = paths.output_root / "checkpoints" / f"agentic_grpo_smoke_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "toy_policy.pt"
    torch.save(
        {
            "parameter": parameter.detach().cpu(),
            "config": grpo_cfg.__dict__,
            "dataset_split": split,
            "num_samples": num_samples,
            "num_generations": num_generations,
        },
        ckpt_path,
    )
    summary = {
        "train_smoke": True,
        "real_model_loaded": False,
        "dataset": summarize_rlvr_dataset(dataset, max_items=2),
        "split": split,
        "num_samples": num_samples,
        "num_generations": num_generations,
        "train_steps": max_steps,
        "rewards": rewards,
        "advantages": [float(x) for x in advantages.tolist()],
        "mean_reward": float(rewards_t.mean().item()),
        "loss_history": loss_history,
        "checkpoint_path": str(ckpt_path),
    }
    log_path = paths.output_root / "logs" / f"train_smoke_agentic_grpo_{run_id}.json"
    latest_path = paths.output_root / "logs" / "train_smoke_agentic_grpo_latest.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["log_path"] = str(log_path)
    summary["latest_log_path"] = str(latest_path)
    return summary

def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Agentic GRPO utilities")
    parser.add_argument("--config", default="configs/agentic_grpo.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run toy rollout + reward + loss + backward")
    parser.add_argument("--train-smoke", action="store_true", help="Run tiny real-data GRPO optimizer smoke test")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    if args.dry_run and args.train_smoke:
        raise SystemExit("Choose only one of --dry-run or --train-smoke")
    if args.dry_run:
        summary = run_dry_run(config_path=args.config, output_root=args.output_root)
    elif args.train_smoke:
        summary = run_train_smoke(config_path=args.config, output_root=args.output_root)
    else:
        summary = run_train_smoke(config_path=args.config, output_root=args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path

from src.rl.grpo_trainer import run_dry_run


def test_run_dry_run_completes_rollout_loss_and_backward(tmp_path):
    summary = run_dry_run(output_root=str(tmp_path / "outputs"))

    assert summary["dry_run"] is True
    assert summary["num_samples"] == 2
    assert summary["num_generations"] == 2
    assert summary["num_trajectories"] == 4
    assert summary["masked_token_count"] > 0
    assert summary["grad_norm"] > 0
    assert summary["valid_json_rate"] == 1.0
    assert (tmp_path / "outputs" / "logs" / "dry_run_agentic_grpo.json").is_file()


def test_run_dry_run_does_not_overwrite_existing_log(tmp_path):
    output_root = tmp_path / "outputs"
    fixed_log = output_root / "logs" / "dry_run_agentic_grpo.json"
    fixed_log.parent.mkdir(parents=True)
    fixed_log.write_text("sentinel", encoding="utf-8")

    summary = run_dry_run(output_root=str(output_root))

    actual_log = Path(summary["log_path"])
    assert fixed_log.read_text(encoding="utf-8") == "sentinel"
    assert actual_log.is_file()
    assert actual_log != fixed_log
    assert actual_log.name.startswith("dry_run_agentic_grpo_")

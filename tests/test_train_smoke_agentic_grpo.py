import json
from pathlib import Path

from PIL import Image

from src.rl.grpo_trainer import run_train_smoke


def test_run_train_smoke_uses_real_adapter_and_saves_outputs(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    raw_img_dir = data_root / "STAR" / "STAR-object" / "train" / "trainimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    raw_img_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(raw_img_dir / "0000.png")
    row = {
        "image_id": 0,
        "width": 16,
        "height": 16,
        "objects": json.dumps([{"id": "ship.1", "bbox": [1, 2, 3, 4]}]),
        "relationships": json.dumps([{"subject": "ship.1", "predicate": "near", "object": "dock.2"}]),
    }
    (jsonl_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "rlvr_dataset:",
                "  source: jsonl_closed",
                f"  jsonl_dir: {jsonl_dir}",
                "  train_split: train",
                "  require_image_exists: true",
                "grpo:",
                "  smoke_num_samples: 1",
                "  smoke_num_generations: 2",
                "  smoke_train_steps: 1",
                "  smoke_seq_len: 4",
                "  learning_rate: 0.001",
                "  clip_eps: 0.2",
                "reward:",
                "  lambda_format: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))

    summary = run_train_smoke(config_path=str(config), output_root=str(output_root))

    assert summary["train_smoke"] is True
    assert summary["real_model_loaded"] is False
    assert summary["num_samples"] == 1
    assert summary["num_generations"] == 2
    assert summary["train_steps"] == 1
    assert summary["loss_history"][0]["grad_norm"] > 0
    assert Path(summary["checkpoint_path"]).is_file()
    assert Path(summary["log_path"]).is_file()
    assert (output_root / "logs" / "train_smoke_agentic_grpo_latest.json").is_file()


def test_run_train_smoke_does_not_overwrite_existing_latest_log(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    raw_img_dir = data_root / "STAR" / "STAR-object" / "train" / "trainimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    raw_img_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(raw_img_dir / "0000.png")
    row = {
        "image_id": 0,
        "width": 16,
        "height": 16,
        "objects": json.dumps([{"id": "ship.1", "bbox": [1, 2, 3, 4]}]),
        "relationships": json.dumps([{"subject": "ship.1", "predicate": "near", "object": "dock.2"}]),
    }
    (jsonl_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    latest_log = output_root / "logs" / "train_smoke_agentic_grpo_latest.json"
    latest_log.parent.mkdir(parents=True)
    latest_log.write_text("sentinel", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "rlvr_dataset:",
                "  source: jsonl_closed",
                f"  jsonl_dir: {jsonl_dir}",
                "  train_split: train",
                "  require_image_exists: true",
                "grpo:",
                "  smoke_num_samples: 1",
                "  smoke_num_generations: 2",
                "  smoke_train_steps: 1",
                "  smoke_seq_len: 4",
                "reward:",
                "  lambda_format: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))

    summary = run_train_smoke(config_path=str(config), output_root=str(output_root))

    assert latest_log.read_text(encoding="utf-8") == "sentinel"
    assert Path(summary["log_path"]).is_file()
    assert summary["latest_log_path"] != str(latest_log)
    assert summary["latest_log_skipped"] is True
    assert Path(summary["latest_log_path"]).is_file()

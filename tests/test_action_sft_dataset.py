import json
from pathlib import Path

from PIL import Image

from src.data.action_sft_dataset import build_action_sft_dataset, sample_to_action_sft_record


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_sample_to_action_sft_record_builds_final_answer_json(tmp_path):
    image_path = tmp_path / "0001.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    sample = {
        "id": "sample-1",
        "image_id": 1,
        "image": str(image_path),
        "prompt": "Generate graph",
        "width": 16,
        "height": 16,
        "objects": [{"id": "ship.1", "bbox": [1, 2, 8, 8]}],
        "relationships": [{"subject": "ship.1", "predicate": "near", "object": "dock.1"}],
    }

    record = sample_to_action_sft_record(sample)
    target = json.loads(record["target_action"])

    assert record["messages"][1]["content"][0]["type"] == "image"
    assert record["messages"][2]["role"] == "assistant"
    assert target["action"] == "final_answer"
    assert target["answer"]["objects"][0]["id"] == "ship.1"
    assert record["conversations"][2]["from"] == "gpt"


def test_build_action_sft_dataset_writes_jsonl_and_hf_dataset(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    image_dir = data_root / "STAR" / "STAR-object" / "train" / "trainimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    image_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(image_dir / "0001.png")
    _write_jsonl(
        jsonl_dir / "train.jsonl",
        [
            {
                "image_id": 1,
                "width": 16,
                "height": 16,
                "objects": json.dumps([{"id": "ship.1", "bbox": [1, 2, 8, 8]}]),
                "relationships": json.dumps([{"subject": "ship.1", "predicate": "near", "object": "dock.1"}]),
            }
        ],
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "rlvr_dataset:",
                "  source: jsonl_closed",
                f"  jsonl_dir: {jsonl_dir}",
                "  input_style: eagle_grounding",
                "  require_image_exists: true",
                "action_sft:",
                "  split: train",
                "  max_samples: 1",
                "  prompt_mode: dataset",
                "  save_hf_dataset: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "outputs"))

    summary = build_action_sft_dataset(config_path=str(config))

    assert summary["num_records"] == 1
    assert Path(summary["jsonl_path"]).is_file()
    assert Path(summary["hf_dataset_path"]).is_dir()
    row = json.loads(Path(summary["jsonl_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][2]["role"] == "assistant"
    assert json.loads(row["target_action"])["action"] == "final_answer"

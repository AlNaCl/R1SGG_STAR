import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.rl.model_load_smoke import (
    ModelLoadSmokeConfig,
    resolve_processor_path,
    run_model_load_smoke,
    sample_to_qwen_messages,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_sample_to_qwen_messages_uses_image_then_text():
    sample = {"image": "/tmp/example.png", "prompt": "Generate graph"}

    messages = sample_to_qwen_messages(sample)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"][0] == {"type": "image", "image": "/tmp/example.png"}
    assert messages[1]["content"][1] == {"type": "text", "text": "Generate graph"}


def test_resolve_processor_path_prefers_model_preprocessor(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    assert resolve_processor_path(str(model_dir)) == str(model_dir.resolve())


def test_run_model_load_smoke_builds_inputs_and_writes_non_overwriting_log(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    image_dir = data_root / "STAR" / "STAR-object" / "val" / "valimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    image_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(image_dir / "0001.png")
    _write_jsonl(
        jsonl_dir / "val.jsonl",
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
    output_root = tmp_path / "outputs"
    fixed_log = output_root / "logs" / "model_load_smoke.json"
    fixed_log.parent.mkdir(parents=True)
    fixed_log.write_text("sentinel", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "rlvr_dataset:",
                "  source: jsonl_closed",
                f"  jsonl_dir: {jsonl_dir}",
                "  input_style: eagle_grounding",
                "  require_image_exists: true",
                "model_load_smoke:",
                "  split: val",
                "  sample_index: 0",
                "  load_model: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages[-1]["content"][0]["type"] == "image"
            return "templated prompt"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["templated prompt"]
            assert len(kwargs["images"]) == 1
            return {"input_ids": torch.ones((1, 3), dtype=torch.long)}

    def fake_processor_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir.resolve())
        return FakeProcessor()

    def fake_model_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir)
        return SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"), eval=lambda: None)

    summary = run_model_load_smoke(
        config_path=str(config),
        output_root=str(output_root),
        model_path=str(model_dir),
        processor_loader=fake_processor_loader,
        model_loader=fake_model_loader,
        vision_info_fn=lambda messages: ([messages[-1]["content"][0]["image"]], None),
    )

    assert fixed_log.read_text(encoding="utf-8") == "sentinel"
    assert summary["model_load_smoke"] is True
    assert summary["model_loaded"] is True
    assert summary["built_processor_inputs"] is True
    assert summary["checkpoint_written"] is False
    assert summary["inputs"]["batch_shapes"]["input_ids"] == [1, 3]
    assert summary["messages"]["user_content_types"] == ["image", "text"]
    assert Path(summary["log_path"]).is_file()
    assert Path(summary["log_path"]) != fixed_log

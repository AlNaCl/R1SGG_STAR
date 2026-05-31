import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.rl.generation_smoke import run_generation_smoke
from src.rl.model_load_smoke import ModelLoadSmokeConfig


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class FakeProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "templated prompt " + str(len(messages))

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        image_count = len(kwargs.get("images") or [])
        return {"input_ids": torch.ones((1, 3 + image_count), dtype=torch.long)}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        token = int(token_ids[0, 0].item())
        if token == 101:
            return ['{"thought":"inspect target","action":"zoom_in","bbox":[0,0,8,8]}']
        if token == 102:
            answer = {
                "objects": [{"id": "ship.1", "bbox": [1, 2, 8, 8]}],
                "relationships": [{"subject": "ship.1", "predicate": "near", "object": "dock.1"}],
            }
            return [json.dumps({"thought": "done", "action": "final_answer", "answer": answer})]
        return ["not json"]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(model_type="fake_qwen_vl")

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls += 1
        input_ids = kwargs["input_ids"]
        token = 101 if self.calls == 1 else 102
        return torch.cat([input_ids, torch.tensor([[token]], dtype=torch.long)], dim=1)


def test_run_generation_smoke_executes_one_zoom_then_final_answer(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    image_dir = data_root / "STAR" / "STAR-object" / "val" / "valimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    image_dir.mkdir(parents=True)
    jsonl_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(image_dir / "0001.png")
    row = {
        "image_id": 1,
        "width": 16,
        "height": 16,
        "objects": json.dumps([{"id": "ship.1", "bbox": [1, 2, 8, 8]}]),
        "relationships": json.dumps([{"subject": "ship.1", "predicate": "near", "object": "dock.1"}]),
    }
    _write_jsonl(jsonl_dir / "val.jsonl", [row])
    output_root = tmp_path / "outputs"
    fixed_log = output_root / "logs" / "generation_smoke.json"
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
                "generation_smoke:",
                "  max_new_tokens: 32",
                "  max_zoom_steps: 1",
                "  coord_type: pixel",
                "  zoom_output_size: 8",
                "reward:",
                "  strict_format: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))

    processor = FakeProcessor()
    model = FakeModel()

    def processor_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir.resolve())
        return processor

    def model_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir)
        return model

    def vision_info(messages):
        images = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(item["image"] for item in content if item.get("type") == "image")
        return images, None

    summary = run_generation_smoke(
        config_path=str(config),
        output_root=str(output_root),
        model_path=str(model_dir),
        processor_loader=processor_loader,
        model_loader=model_loader,
        vision_info_fn=vision_info,
    )

    assert fixed_log.read_text(encoding="utf-8") == "sentinel"
    assert summary["generation_smoke"] is True
    assert summary["no_training"] is True
    assert summary["checkpoint_written"] is False
    assert summary["trajectory"]["used_zoom"] is True
    assert summary["trajectory"]["num_steps"] == 2
    assert summary["trajectory"]["steps"][0]["action"] == "zoom_in"
    assert summary["trajectory"]["steps"][0]["observation"]["valid"] is True
    assert summary["trajectory"]["steps"][1]["action"] == "final_answer"
    assert summary["reward"]["is_valid_json"] is True
    assert Path(summary["log_path"]).is_file()
    assert Path(summary["log_path"]) != fixed_log

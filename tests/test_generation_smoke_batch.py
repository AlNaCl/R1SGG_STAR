import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.rl.generation_smoke_batch import run_generation_batch_smoke
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
        return "templated prompt"

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": torch.ones((1, 3), dtype=torch.long)}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        token = int(token_ids[0, 0].item())
        if token == 201:
            return ['{"thought":"done","action":"final_answer","answer":{"objects":[],"relationships":[]}}']
        if token == 202:
            return ['{"thought":"done","action":"final_answer","answer":{"objects":[],"relationships":[]}} extra']
        return ["not json"]


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, token):
        self.token = token
        self.config = SimpleNamespace(model_type="fake_qwen_vl")

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, torch.tensor([[self.token]], dtype=torch.long)], dim=1)


def test_run_generation_batch_smoke_compares_base_and_adapter(tmp_path, monkeypatch):
    data_root = tmp_path / "STAR"
    image_dir = data_root / "STAR" / "STAR-object" / "val" / "valimg正确"
    jsonl_dir = data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    image_dir.mkdir(parents=True)
    rows = []
    for image_id in (1, 2):
        Image.new("RGB", (16, 16), "white").save(image_dir / f"{image_id:04d}.png")
        rows.append(
            {
                "image_id": image_id,
                "width": 16,
                "height": 16,
                "objects": json.dumps([]),
                "relationships": json.dumps([]),
            }
        )
    _write_jsonl(jsonl_dir / "val.jsonl", rows)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_root = tmp_path / "outputs"
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
                "  prompt_mode: action_only",
                "reward:",
                "  strict_format: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))

    def processor_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir.resolve())
        return FakeProcessor()

    def model_loader(path, cfg: ModelLoadSmokeConfig):
        assert path == str(model_dir)
        return FakeModel(202 if cfg.peft_adapter_path else 201)

    def vision_info(messages):
        images = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                images.extend(item["image"] for item in content if item.get("type") == "image")
        return images, None

    summary = run_generation_batch_smoke(
        config_path=str(config),
        output_root=str(output_root),
        model_path=str(model_dir),
        peft_adapter_path=str(adapter_dir),
        num_samples=2,
        processor_loader=processor_loader,
        model_loader=model_loader,
        vision_info_fn=vision_info,
    )

    assert summary["generation_batch_smoke"] is True
    assert summary["sample_indices"] == [0, 1]
    assert summary["variants"]["base"]["metrics"]["valid_json_rate"] == 1.0
    assert summary["variants"]["base"]["metrics"]["final_answer_valid_rate"] == 1.0
    assert summary["variants"]["adapter"]["metrics"]["valid_json_rate"] == 0.0
    assert summary["variants"]["adapter"]["metrics"]["extra_text_rate"] == 1.0
    assert summary["comparison"]["adapter_minus_base"]["valid_json_rate"] == -1.0
    assert Path(summary["log_path"]).is_file()

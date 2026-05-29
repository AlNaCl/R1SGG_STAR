import json

from PIL import Image

from src.data.rlvr_dataset import (
    JsonlRLVRDataset,
    RLVRDatasetConfig,
    load_rlvr_dataset,
    summarize_rlvr_dataset,
)
from src.rl.paths import AgenticPaths


def make_paths(tmp_path):
    data_root = tmp_path / "STARROOT"
    return AgenticPaths(
        data_root=data_root,
        r1sgg_data_root=data_root / "r1sgg_data",
        star_raw_root=data_root / "STAR",
        output_root=tmp_path / "outputs",
        dataset_path=data_root / "r1sgg_data" / "star_r1sgg_hf_closed",
        jsonl_closed_dir=data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed",
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_jsonl_adapter_normalizes_closed_star_row_with_image_field(tmp_path):
    image_path = tmp_path / "0001.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    jsonl_path = tmp_path / "train.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {
                "image_id": 1,
                "width": 32,
                "height": 24,
                "image": str(image_path),
                "prompt_close": "Generate graph",
                "objects": json.dumps([{"id": "ship.1", "bbox": [1, 2, 3, 4]}]),
                "relationships": json.dumps([{"subject": "ship.1", "predicate": "near", "object": "dock.2"}]),
            }
        ],
    )

    ds = JsonlRLVRDataset(jsonl_path, split="train", paths=make_paths(tmp_path))
    sample = ds[0]

    assert len(ds) == 1
    assert sample["id"] == "star_train_1"
    assert sample["task_type"] == "scene_graph"
    assert sample["image"] == str(image_path)
    assert sample["prompt"] == "Generate graph"
    assert sample["answer"]["objects"] == sample["objects"]
    assert sample["relationships"][0]["predicate"] == "near"


def test_jsonl_adapter_maps_missing_image_from_star_split_dirs(tmp_path):
    paths = make_paths(tmp_path)
    image_dir = paths.star_raw_root / "STAR-object" / "val" / "valimg正确"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "0002.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    jsonl_path = tmp_path / "val.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {
                "image_id": 2,
                "width": 16,
                "height": 16,
                "objects": [],
                "relationships": [],
            }
        ],
    )

    ds = JsonlRLVRDataset(jsonl_path, split="val", paths=paths)

    assert ds[0]["image"] == str(image_path.resolve())


def test_jsonl_adapter_can_skip_image_existence_check(tmp_path):
    jsonl_path = tmp_path / "test.jsonl"
    write_jsonl(
        jsonl_path,
        [{"image_id": 9, "width": 1, "height": 1, "objects": "[]", "relationships": "[]"}],
    )

    ds = JsonlRLVRDataset(
        jsonl_path,
        split="test",
        paths=make_paths(tmp_path),
        require_image_exists=False,
    )

    assert ds[0]["image"].endswith("STAR-object/test/testimg正确/0009.png")


def test_load_rlvr_dataset_selects_jsonl_closed_source(tmp_path):
    paths = make_paths(tmp_path)
    image_dir = paths.star_raw_root / "STAR-object" / "train" / "trainimg正确"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_dir / "0003.png")
    write_jsonl(
        paths.jsonl_closed_dir / "train.jsonl",
        [{"image_id": 3, "width": 8, "height": 8, "objects": [], "relationships": []}],
    )

    ds = load_rlvr_dataset(
        RLVRDatasetConfig(source="jsonl_closed", split="train"),
        paths=paths,
    )

    assert len(ds) == 1
    assert ds[0]["image_id"] == 3


def test_summarize_rlvr_dataset_reports_counts(tmp_path):
    image_path = tmp_path / "0001.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    jsonl_path = tmp_path / "train.jsonl"
    write_jsonl(
        jsonl_path,
        [
            {
                "image_id": 1,
                "width": 32,
                "height": 24,
                "image": str(image_path),
                "objects": [{"id": "ship.1"}],
                "relationships": [{"subject": "ship.1", "predicate": "near", "object": "dock.2"}],
            }
        ],
    )
    ds = JsonlRLVRDataset(jsonl_path, split="train", paths=make_paths(tmp_path))

    summary = summarize_rlvr_dataset(ds)

    assert summary["num_rows"] == 1
    assert summary["examples"][0]["num_objects"] == 1
    assert summary["examples"][0]["num_relationships"] == 1

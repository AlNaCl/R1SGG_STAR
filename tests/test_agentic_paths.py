from src.rl.paths import ensure_output_dirs, resolve_agentic_paths


def test_resolve_agentic_paths_uses_env(monkeypatch, tmp_path):
    data_root = tmp_path / "STAR"
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("R1SGG_DATA_ROOT", str(data_root / "r1sgg_data"))
    monkeypatch.setenv("STAR_RAW_ROOT", str(data_root / "STAR"))
    monkeypatch.setenv("OUTPUT_ROOT", str(output_root))

    paths = resolve_agentic_paths()

    assert paths.dataset_path == data_root / "r1sgg_data" / "star_r1sgg_hf_closed"
    assert paths.jsonl_closed_dir == data_root / "r1sgg_data" / "star_r1sgg_jsonl_closed"
    assert paths.output_root == output_root


def test_ensure_output_dirs_creates_expected_subdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "agentic_outputs"))

    subdirs = ensure_output_dirs()

    assert set(subdirs) == {"logs", "checkpoints", "predictions", "eval_results", "tmp"}
    assert all(path.is_dir() for path in subdirs.values())

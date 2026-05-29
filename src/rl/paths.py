"""Path handling for Agentic GRPO / RLVR experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/STAR")
DEFAULT_R1SGG_DATA_ROOT = DEFAULT_DATA_ROOT / "r1sgg_data"
DEFAULT_STAR_RAW_ROOT = DEFAULT_DATA_ROOT / "STAR"
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs")
OUTPUT_SUBDIRS = ("logs", "checkpoints", "predictions", "eval_results", "tmp")


@dataclass(frozen=True)
class AgenticPaths:
    """Resolved filesystem paths used by the Agentic GRPO pipeline."""

    data_root: Path
    r1sgg_data_root: Path
    star_raw_root: Path
    output_root: Path
    dataset_path: Path
    jsonl_closed_dir: Path

    @property
    def output_subdirs(self) -> dict[str, Path]:
        return {name: self.output_root / name for name in OUTPUT_SUBDIRS}


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def resolve_agentic_paths(create_output: bool = False) -> AgenticPaths:
    """Resolve standard data/output paths from env vars with project defaults."""

    data_root = _env_path("DATA_ROOT", DEFAULT_DATA_ROOT)
    r1sgg_data_root = _env_path("R1SGG_DATA_ROOT", data_root / "r1sgg_data")
    star_raw_root = _env_path("STAR_RAW_ROOT", data_root / "STAR")
    output_root = _env_path("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    paths = AgenticPaths(
        data_root=data_root,
        r1sgg_data_root=r1sgg_data_root,
        star_raw_root=star_raw_root,
        output_root=output_root,
        dataset_path=r1sgg_data_root / "star_r1sgg_hf_closed",
        jsonl_closed_dir=r1sgg_data_root / "star_r1sgg_jsonl_closed",
    )
    if create_output:
        ensure_output_dirs(paths)
    return paths


def ensure_output_dirs(paths: AgenticPaths | None = None) -> dict[str, Path]:
    """Create and return the standard output subdirectories."""

    resolved = paths or resolve_agentic_paths(create_output=False)
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    for subdir in resolved.output_subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    return resolved.output_subdirs

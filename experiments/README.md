# Experiments Management

This folder stores lightweight, reproducible experiment records.

## Naming convention

- Per-run folder: `YYYY-MM-DD_topic_vN`
- Example: `2026-04-17_geom_prompt_v1`

## Recommended files per run

- `config.yaml`: full runnable config snapshot
- `metrics.json`: final key metrics
- `notes.md`: goal, change, result, conclusion, next
- `run.log`: raw logs (ignored by git via `.gitignore`)

## Daily fixed workflow

1. Create branch from `dev`: `exp/<topic>`
2. Implement and commit code changes in small chunks
3. Commit once right before running the experiment
4. Record commit id in `notes.md`
5. Run and save outputs under one run folder
6. Update `experiments/summary.csv`
7. Commit lightweight records and push branch

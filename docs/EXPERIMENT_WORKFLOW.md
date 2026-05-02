# Branch and Experiment Workflow

## Branch roles

- `main`: stable, verified code only
- `dev`: active development and integration
- `exp/<topic>`: one branch for one experiment theme

## Merge policy

- Merge `exp/<topic>` -> `dev` when changes are useful and non-breaking
- Merge `dev` -> `main` only when stable and reproducible

## Commit style

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code cleanup without behavior change
- `exp:` experiment-prep code snapshot
- `docs:` records and documentation updates

## Minimal one-day checklist

1. `git checkout dev && git pull origin dev`
2. `git checkout -b exp/<topic>`
3. Code + small commits
4. Pre-run snapshot commit: `exp: prepare <topic> vN`
5. Save commit id: `git rev-parse --short HEAD`
6. Run experiment, save outputs under `experiments/YYYY-MM-DD_topic_vN/`
7. Update `notes.md` and `experiments/summary.csv`
8. Commit records and push experiment branch

## Suggested experiment folder layout

```
experiments/YYYY-MM-DD_topic_vN/
  config.yaml
  metrics.json
  notes.md
  run.log
```

# AGENTS.md

## Project Role

You are working in a remote sensing scene graph generation / vision-language model research repository.

The current goal is to implement an Agentic GRPO / RLVR training pipeline inspired by the paper:

**Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding**

The implementation should be practical, incremental, and compatible with the existing R1-SGG / STAR project structure.

Do not rewrite the whole repository. Prefer small, testable, and reversible changes.

---

## Runtime Environment

The active conda environment is:

```bash
/root/miniconda3/envs/r1sgg
```

Before running any Python command, activate the environment with:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg
```

Do not create a new conda environment unless explicitly requested.

Prefer using the existing `r1sgg` environment and existing installed packages.

If a dependency is missing, report it first. Do not install large packages or change CUDA / PyTorch / flash-attn versions without confirmation.

---

## Data and Storage Paths

The main dataset root is:

```bash
/root/autodl-tmp/STAR
```

Current important directories are:

```text
/root/autodl-tmp/STAR/
├── r1sgg_data/
└── STAR/
```

The output root for this Agentic GRPO / RLVR experiment should be newly created as:

```bash
/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs
```

Use the following environment variables in scripts and configs:

```bash
export DATA_ROOT=/root/autodl-tmp/STAR
export R1SGG_DATA_ROOT=/root/autodl-tmp/STAR/r1sgg_data
export STAR_RAW_ROOT=/root/autodl-tmp/STAR/STAR
export OUTPUT_ROOT=/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs

export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_ext
export TRITON_CACHE_DIR=/root/autodl-tmp/triton_cache
```

Path rules:

* Use `/root/autodl-tmp/STAR` as the dataset root.
* Use `/root/autodl-tmp/STAR/r1sgg_data` for R1-SGG-style processed data.
* Use `/root/autodl-tmp/STAR/STAR` for original STAR files if needed.
* Use `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs` for this experiment’s logs, checkpoints, predictions, temporary files, and evaluation results.
* Do not write large datasets, model weights, checkpoints, generated predictions, or temporary image crops inside the Git repository.
* Do not overwrite existing outputs under `/root/autodl-tmp/STAR_SGG_QWen3_outputs`.
* Before assuming the exact data format, inspect the directory structure under `/root/autodl-tmp/STAR`.

Recommended output structure:

```text
/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/
├── logs/
├── checkpoints/
├── predictions/
├── eval_results/
└── tmp/
```

Create it with:

```bash
mkdir -p /root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/{logs,checkpoints,predictions,eval_results,tmp}
```

---

## Development Rules

### General

* Prefer modifying existing modules over creating parallel duplicated systems.
* Avoid large refactors unless necessary.
* Keep changes small and testable.
* Use type hints for new Python functions.
* Add docstrings for important functions.
* Do not hard-code absolute paths in Python code when environment variables or config values can be used.
* Do not hard-code GPU count.
* Do not assume data format before inspecting files.
* Do not silently swallow exceptions. Log clear error messages.

### Large Remote Sensing Images

* Avoid unnecessary full-image copies.
* Avoid loading all images into memory at once.
* Use lazy loading where possible.
* Keep temporary crops under `OUTPUT_ROOT/tmp` if they need to be materialized.
* Prefer in-memory PIL crops for short-lived tool observations.

### Training and Evaluation

* Reuse existing training, inference, and evaluation utilities when possible.
* When adding new configs, keep paths configurable.
* When adding new scripts, support environment variables.
* When adding reward or evaluation logic, preserve compatibility with future remote sensing SGG metrics such as R@K, mR@K, MR@K, mMR@K, and HMR@K.
* Always provide a dry-run mode with tiny toy data before real training.

### Tool Observation Masking

For Agentic RLVR / Agentic GRPO:

* Prompt tokens must not contribute to loss.
* Image tokens must not contribute to loss.
* Tool observation tokens must not contribute to loss.
* Only model-generated text/action/final-answer tokens should contribute to policy loss.
* Zoom-in crop images or crop descriptions are conditioning observations, not supervised targets.

---

## Testing Rules

Before finishing a phase, run the most relevant tests.

If tests already exist, reuse the existing test framework.

Recommended tests for this task:

```bash
pytest tests/test_zoom_tool.py -q
pytest tests/test_reward.py -q
pytest tests/test_token_mask.py -q
pytest tests/test_grpo_loss.py -q
```

For early phases, run only the tests related to the modified module.

Every final response after a coding phase should include:

1. Modified or added files.
2. What each file does.
3. Commands that were run.
4. Test results.
5. Any assumptions about data format.
6. Remaining limitations or questions.

---

## Phase Discipline

Do not implement everything at once.

Follow this order unless explicitly told otherwise:

```text
Phase 0: Inspect repository and data structure.
Phase 1: Create config, path handling, output directories, and zoom-in tool.
Phase 2: Implement reward functions.
Phase 3: Implement token-level loss masking.
Phase 4: Implement agentic rollout.
Phase 5: Implement GRPO trainer.
Phase 6: Implement dry-run script.
Phase 7: Connect real STAR / R1SGG data.
Phase 8: Connect real model training.
Phase 9: Add evaluation and Pass@K.
```

At the end of each phase, stop and report progress. Do not proceed to the next phase without confirmation.

---

## First Required Action

Before implementing Agentic GRPO code, inspect the real dataset structure:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

echo "DATA_ROOT=$DATA_ROOT"
echo "R1SGG_DATA_ROOT=$R1SGG_DATA_ROOT"
echo "STAR_RAW_ROOT=$STAR_RAW_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"

mkdir -p /root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/{logs,checkpoints,predictions,eval_results,tmp}

ls -R /root/autodl-tmp/STAR | head -200
find /root/autodl-tmp/STAR -maxdepth 3 -type f | head -100
```

Then report:

1. The detected dataset directory structure.
2. Which files look like training annotations.
3. Which files look like image files.
4. Which files look like evaluation annotations.
5. Any unclear points that need user confirmation.

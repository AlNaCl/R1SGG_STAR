# Agentic GRPO / RLVR Implementation Spec

## 1. Goal

Implement an Agentic GRPO / RLVR pipeline for the current R1-SGG / STAR remote sensing project.

The implementation should be inspired by the method in:

**Text Before Vision: Staged Knowledge Injection Matters for Agentic RLVR in Ultra-High-Resolution Remote Sensing Understanding**

The target method is not plain GRPO. It is:

```text
Cold-start SFT with domain knowledge
+ hard UHR image-text pre-warming
+ zoom-in tool based Agentic RLVR
+ GRPO optimization
+ verifiable reward
+ token-level masking for tool observations
```

The implementation should be incremental and compatible with the existing repository.

---

## 2. Real Server Environment

Use the existing conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg
```

Environment path:

```bash
/root/miniconda3/envs/r1sgg
```

Do not create a new environment.

---

## 3. Real Data and Output Paths

The real dataset root is:

```bash
/root/autodl-tmp/STAR
```

Important existing directories:

```text
/root/autodl-tmp/STAR/r1sgg_data
/root/autodl-tmp/STAR/STAR
```

For this new Agentic GRPO / RLVR experiment, use this new output root:

```bash
/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs
```

Do not use or overwrite:

```bash
/root/autodl-tmp/STAR_SGG_QWen3_outputs
```

Use these environment variables:

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

Create output directories if needed:

```bash
mkdir -p /root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/{logs,checkpoints,predictions,eval_results,tmp}
```

---

## 4. Required First Step: Inspect Real Data

Before writing the main GRPO code, inspect the dataset and existing project structure.

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export DATA_ROOT=/root/autodl-tmp/STAR
export R1SGG_DATA_ROOT=/root/autodl-tmp/STAR/r1sgg_data
export STAR_RAW_ROOT=/root/autodl-tmp/STAR/STAR
export OUTPUT_ROOT=/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs

mkdir -p ${OUTPUT_ROOT}/{logs,checkpoints,predictions,eval_results,tmp}

pwd
ls
find . -maxdepth 3 -type f | sed 's#^\./##' | head -200

echo "===== STAR DATA STRUCTURE ====="
ls -R ${DATA_ROOT} | head -200

echo "===== POSSIBLE ANNOTATION FILES ====="
find ${DATA_ROOT} -maxdepth 5 -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.pkl" -o -name "*.txt" -o -name "*.csv" \) | head -200

echo "===== POSSIBLE IMAGE FILES ====="
find ${DATA_ROOT} -maxdepth 5 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) | head -100
```

Then report:

1. The repository structure.
2. The STAR dataset structure.
3. Which files look like annotations.
4. Which files look like images.
5. Which files are likely already processed for R1-SGG.
6. What assumptions need confirmation.

Stop after this inspection unless explicitly told to continue.

---

## 5. Target Repository Additions

Recommended new files:

```text
configs/agentic_grpo.yaml

src/tools/zoom_tool.py

src/rl/rewards.py
src/rl/mask_utils.py
src/rl/agentic_rollout.py
src/rl/grpo_trainer.py

src/data/rlvr_dataset.py

scripts/train_sft_text_before_vision.sh
scripts/train_agentic_grpo.sh
scripts/dry_run_agentic_grpo.sh

tests/test_zoom_tool.py
tests/test_reward.py
tests/test_token_mask.py
tests/test_grpo_loss.py
```

If the repository already has equivalent modules, extend existing modules instead of duplicating them.

---

## 6. Method Overview

Implement a two-stage training recipe.

### Stage 1: Cold-start SFT

SFT data should include:

1. Earth-science text QA with CoT.
2. UHR remote sensing image-text QA / STAR / SuperRS-VQA-style samples.

Purpose:

* Inject domain concepts.
* Inject Earth-science mechanisms.
* Inject remote sensing reasoning rules.
* Pre-warm the model on hard UHR image-text examples.
* Reduce blind exploration during later RL.

### Stage 2: Agentic GRPO / RLVR

RL data should include:

1. General verifiable-reward data if available.
2. Hard UHR RS image-text samples.
3. Eventually STAR / R1-SGG-style samples.

Purpose:

* Train the model to actively acquire visual evidence.
* Use a zoom-in tool to inspect local regions.
* Optimize answer correctness with verifiable reward.
* Use GRPO group-relative advantage instead of a separate value model.

---

## 7. Agentic Rollout Format

Implement `src/rl/agentic_rollout.py`.

The agentic state is:

```text
z_k = [(u_0, v_0), (u_1, v_1), ..., (u_k, v_k)]
```

Where:

* `u_k` is model-generated text, reasoning, tool action, or final answer.
* `v_k` is external tool observation, such as zoom-in crop image, crop metadata, or error message.
* Tool observations are conditioning context only and must not contribute to LM loss.

Rollout logic:

```python
for each sample:
    history = prompt + global_image

    for step in range(max_tool_steps):
        model generates JSON

        if action == "zoom_in":
            parse bbox
            call zoom_in(image, bbox)
            append tool observation to history
        elif action == "final_answer":
            stop rollout
        else:
            mark invalid action
            stop or continue according to config

    compute reward
    return trajectory, token masks, reward
```

---

## 8. Required Model Output Format

Constrain model output to JSON-style actions.

### Zoom-in Action

```json
{
  "thought": "brief reason for zooming into this region",
  "action": "zoom_in",
  "bbox": [x1, y1, x2, y2]
}
```

### Final Answer

```json
{
  "thought": "brief reasoning process",
  "action": "final_answer",
  "answer": "final answer here"
}
```

Requirements:

* Support normalized coordinates `[0, 1]`.
* Support pixel coordinates.
* Coordinate mode should be controlled by config.
* Invalid JSON should not crash training.
* Invalid bbox should not crash training.
* Invalid action should produce low format reward.
* `max_tool_steps` should be configurable.
* `num_generations` should be configurable for GRPO group sampling.

---

## 9. Zoom-in Tool

Implement `src/tools/zoom_tool.py`.

Required function:

```python
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from PIL import Image

def zoom_in(
    image: Union[str, Image.Image],
    bbox: List[float],
    coord_type: Literal["normalized", "pixel"] = "normalized",
    output_size: int = 448,
    min_bbox_area_ratio: float = 1e-4,
    max_bbox_area_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Crop and resize a region from a large remote sensing image.

    Returns:
        {
            "crop_image": PIL.Image or None,
            "bbox_pixel": [x1, y1, x2, y2],
            "valid": bool,
            "clipped": bool,
            "error": optional str,
            "area_ratio": float
        }
    """
```

Behavior:

* Accept image path or PIL image.
* Convert normalized bbox to pixel bbox if needed.
* Clip bbox to image boundary.
* Mark `clipped=True` if coordinates were clipped.
* Mark `valid=False` if:

  * bbox length is not 4;
  * x2 <= x1;
  * y2 <= y1;
  * area is too small;
  * area is too large;
  * image cannot be opened.
* Return a resized crop if valid.
* Do not save crop images unless explicitly requested.
* Avoid unnecessary full-image copies.

Tests:

```bash
pytest tests/test_zoom_tool.py -q
```

Test cases:

* normalized bbox works;
* pixel bbox works;
* out-of-bound bbox is clipped;
* invalid bbox does not crash;
* tiny bbox is invalid;
* too-large bbox can be penalized or invalid depending on config.

---

## 10. Reward Design

Implement `src/rl/rewards.py`.

Paper-style reward:

```text
G(γ) = S_ok(γ) + S_fmt(γ) + 1[S_ok(γ)=1] * S_tool(γ)
```

Implement a configurable version:

```python
reward = correctness_reward \
       + lambda_format * format_reward \
       + int(correctness_reward > 0) * lambda_tool * tool_reward
```

### 10.1 Correctness Reward

Implement `compute_correctness_reward`.

Task types:

```text
multiple_choice
true_false
fill_blank
numeric
open_qa
scene_graph
```

Rules:

* Multiple choice: normalized exact match.
* True/False: normalized exact match.
* Fill blank: normalized text or numeric match.
* Numeric: numeric tolerance.
* Open QA: simple normalized text / keyword match first; leave LLM-as-judge interface for later.
* Scene graph: leave interface for triplet matching, predicate matching, MR@K, mMR@K, and HMR@K.

Return:

```python
0.0 or 1.0
```

### 10.2 Format Reward

Implement `compute_format_reward`.

Check:

* JSON parseable.
* Has `action`.
* `action` is one of `zoom_in` or `final_answer`.
* For `zoom_in`, has valid-looking `bbox`.
* For `final_answer`, has non-empty `answer`.
* No extra unparseable text outside JSON if strict mode is enabled.

Return:

```python
float in [0, 1]
```

### 10.3 Tool Reward

Implement `compute_tool_reward`.

Only apply it when answer is correct.

Reward effective evidence acquisition:

Positive factors:

* At least one valid zoom-in call.
* Bbox is valid.
* Bbox is not the whole image.
* Bbox is not extremely tiny.
* Tool calls are within `max_tool_steps`.
* If evidence bbox exists, zoom region overlaps target evidence.

Negative factors:

* Invalid bbox.
* Repeated almost-identical zoom regions.
* Too many tool calls.
* Whole-image zoom.
* Tiny meaningless crop.

Return:

```python
float in [0, 1]
```

### 10.4 Combined Reward

Implement:

```python
def compute_total_reward(
    prediction: str,
    sample: dict,
    trajectory: Optional[dict],
    config: dict,
) -> dict:
    """
    Returns:
        {
            "reward": float,
            "correctness_reward": float,
            "format_reward": float,
            "tool_reward": float,
            "is_correct": bool,
            "is_valid_json": bool,
            "used_zoom": bool,
            "invalid_bbox": bool
        }
    """
```

Tests:

```bash
pytest tests/test_reward.py -q
```

---

## 11. Token-level Loss Masking

Implement `src/rl/mask_utils.py`.

Core rule:

```text
Only model-generated tokens contribute to policy loss.
Prompt tokens, image tokens, and tool observation tokens must be masked out.
```

Implement span-based masking.

Suggested data structure:

```python
@dataclass
class TokenSpan:
    start: int
    end: int
    role: str
```

Roles:

```text
prompt
image
tool_observation
model_generation
```

Mask rule:

```python
loss_mask = torch.zeros_like(input_ids)

for span in spans:
    if span.role == "model_generation":
        loss_mask[span.start:span.end] = 1
    else:
        loss_mask[span.start:span.end] = 0
```

Required function:

```python
def build_loss_mask(input_ids: torch.Tensor, spans: list[TokenSpan]) -> torch.Tensor:
    ...
```

Tests:

```bash
pytest tests/test_token_mask.py -q
```

Test cases:

* prompt mask is 0;
* image span mask is 0;
* tool observation span mask is 0;
* generated action span mask is 1;
* generated answer span mask is 1.

---

## 12. GRPO Trainer

Implement `src/rl/grpo_trainer.py`.

Do not train a separate critic / value model.

For each prompt, sample `G = num_generations` trajectories:

```python
responses = policy.generate(prompt, num_return_sequences=G)
rewards = reward_fn(responses)
advantages = (rewards - rewards.mean()) / (rewards.std() + eps)
```

Compute token-level ratio:

```python
ratio = exp(logprob_new - logprob_old)
```

Clipped GRPO loss:

```python
loss_i = -min(
    ratio * advantage,
    clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
)
```

Masked loss:

```python
loss = masked_mean(loss_i, loss_mask)
```

Optional KL penalty against frozen reference model:

```python
loss = grpo_loss + beta_kl * kl
```

Requirements:

* Cache `old_logprobs` during rollout.
* Freeze `ref_model`.
* Support bf16.
* Support gradient accumulation.
* Support LoRA if project already uses it.
* Support accelerate/deepspeed only if already used by the project.
* Do not hard-code GPU count.
* Make dry-run possible on tiny toy samples.

Log every step:

```text
mean_reward
correctness_reward
format_reward
tool_reward
valid_json_rate
zoom_in_usage_rate
invalid_bbox_rate
mean_trajectory_length
kl
loss
advantage_mean
advantage_std
```

Tests:

```bash
pytest tests/test_grpo_loss.py -q
```

Test cases:

* group rewards are normalized correctly;
* loss can backward;
* no value model is required;
* loss mask is respected.

---

## 13. Config File

Create `configs/agentic_grpo.yaml`.

Initial content:

```yaml
env:
  conda_env: /root/miniconda3/envs/r1sgg

paths:
  data_root: ${DATA_ROOT:/root/autodl-tmp/STAR}
  r1sgg_data_root: ${R1SGG_DATA_ROOT:/root/autodl-tmp/STAR/r1sgg_data}
  star_raw_root: ${STAR_RAW_ROOT:/root/autodl-tmp/STAR/STAR}
  output_root: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}
  tmp_dir: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}/tmp
  log_dir: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}/logs
  checkpoint_dir: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}/checkpoints
  prediction_dir: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}/predictions
  eval_dir: ${OUTPUT_ROOT:/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}/eval_results

model:
  name_or_path: /path/to/sft_checkpoint
  ref_model_name_or_path: /path/to/sft_checkpoint
  dtype: bf16
  gradient_checkpointing: true
  use_lora: false

data:
  rl_data: []
  image_root: /root/autodl-tmp/STAR
  question_key: question
  answer_key: answer
  image_key: image
  task_type_key: task_type
  evidence_bbox_key: evidence_bbox

rollout:
  num_generations: 4
  max_tool_steps: 2
  max_new_tokens: 512
  temperature: 1.0
  top_p: 0.95
  action_format: json
  stop_on_invalid_action: true

tool:
  name: zoom_in
  coord_type: normalized
  crop_output_size: 448
  min_bbox_area_ratio: 0.0001
  max_bbox_area_ratio: 0.8

reward:
  lambda_format: 0.1
  lambda_tool: 0.2
  numeric_tolerance: 1.0e-3
  tool_bonus_only_when_correct: true
  strict_json: true

grpo:
  clip_eps: 0.2
  beta_kl: 0.01
  eps: 1.0e-6
  learning_rate: 1.0e-6
  train_steps: 80
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0

logging:
  log_steps: 1
  save_steps: 20
  report_to: tensorboard

dry_run:
  enabled: false
  num_samples: 2
  num_generations: 2
```

If the repository config system does not support `${VAR:default}` syntax, replace it with normal strings or implement a small environment-variable resolver.

---

## 14. Training Scripts

Create `scripts/train_sft_text_before_vision.sh`.

Purpose:

```text
Earth-science Text QA + STAR / UHR image-text data -> cold-start SFT checkpoint
```

Requirements:

* Use existing SFT pipeline if available.
* Do not implement a new full SFT trainer unless necessary.
* Use environment variables.
* Save checkpoints to `${OUTPUT_ROOT}/checkpoints/sft_text_before_vision`.

Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/tmp"

echo "Implement this script by reusing the existing SFT entrypoint."
echo "DATA_ROOT=${DATA_ROOT}"
echo "R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT}"
echo "STAR_RAW_ROOT=${STAR_RAW_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
```

Create `scripts/train_agentic_grpo.sh`.

Purpose:

```text
SFT checkpoint -> Agentic GRPO with zoom-in tool
```

Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export R1SGG_DATA_ROOT=${R1SGG_DATA_ROOT:-/root/autodl-tmp/STAR/r1sgg_data}
export STAR_RAW_ROOT=${STAR_RAW_ROOT:-/root/autodl-tmp/STAR/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/predictions" "${OUTPUT_ROOT}/eval_results" "${OUTPUT_ROOT}/tmp"

python -m src.rl.grpo_trainer --config "${CONFIG}"
```

Create `scripts/dry_run_agentic_grpo.sh`.

Purpose:

```text
Run toy rollout + reward + loss + backward with 2 toy samples.
```

Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/STAR}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs}

CONFIG=${CONFIG:-configs/agentic_grpo.yaml}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/tmp"

python -m src.rl.grpo_trainer --config "${CONFIG}" --dry-run
```

---

## 15. Data Format Support

Support JSONL first.

Text QA sample:

```json
{
  "id": "es_0001",
  "task_type": "open_qa",
  "question": "What is NDVI and why does it saturate in dense vegetation?",
  "answer": "NDVI is ..."
}
```

Image-text QA sample:

```json
{
  "id": "rs_0001",
  "task_type": "multiple_choice",
  "image": "xxx.png",
  "question": "Which option best describes the object in the highlighted area?",
  "choices": ["A", "B", "C", "D"],
  "answer": "B",
  "evidence_bbox": [0.2, 0.3, 0.4, 0.5]
}
```

Scene graph sample placeholder:

```json
{
  "id": "sgg_0001",
  "task_type": "scene_graph",
  "image": "xxx.png",
  "objects": [],
  "relations": [],
  "answer": []
}
```

Before connecting the real STAR data, inspect the actual annotation format and write an adapter.

---

## 16. Evaluation

Implement or reuse evaluation scripts for:

```text
Pass@1
Pass@K
Pass@32
average reward
correctness rate
valid JSON rate
zoom-in usage rate
invalid bbox rate
average number of tool calls
```

Pass@K:

For each problem, sample `n >= k` responses and count correct samples `c`.

For `k = 1`, report average single-shot accuracy.

For `k > 1`, report whether at least one of the sampled candidates is correct under the fixed budget.

---

## 17. Phase Plan

### Phase 0: Inspection Only

Tasks:

1. Inspect repository.
2. Inspect `/root/autodl-tmp/STAR`.
3. Create output directories.
4. Report likely data files and image files.

Do not implement GRPO yet.

Expected final report:

```text
Modified files: none or AGENTS/config only
Dataset structure:
Likely annotation files:
Likely image files:
Questions for user:
```

### Phase 1: Paths + Config + Zoom Tool

Tasks:

1. Create `configs/agentic_grpo.yaml`.
2. Create `src/tools/zoom_tool.py`.
3. Create `tests/test_zoom_tool.py`.
4. Run zoom tool tests.

Stop after Phase 1.

### Phase 2: Reward

Tasks:

1. Create `src/rl/rewards.py`.
2. Create `tests/test_reward.py`.
3. Implement correctness, format, tool, and total reward.
4. Run reward tests.

Stop after Phase 2.

### Phase 3: Token Mask

Tasks:

1. Create `src/rl/mask_utils.py`.
2. Create `tests/test_token_mask.py`.
3. Implement span-based loss mask.
4. Run token mask tests.

Stop after Phase 3.

### Phase 4: Agentic Rollout

Tasks:

1. Create `src/rl/agentic_rollout.py`.
2. Implement JSON action parsing.
3. Implement zoom-in tool call integration.
4. Implement trajectory records.
5. Ensure tool observations are identifiable for masking.

Stop after Phase 4.

### Phase 5: GRPO Loss and Trainer

Tasks:

1. Create `src/rl/grpo_trainer.py`.
2. Implement group reward normalization.
3. Implement clipped GRPO loss.
4. Implement optional KL against reference model.
5. Implement logging.

Stop after Phase 5.

### Phase 6: Dry Run

Tasks:

1. Create toy data.
2. Run 2 samples.
3. Use 2 generations per sample.
4. Complete rollout, reward, GRPO loss, and backward.
5. Report logs.

Stop after Phase 6.

### Phase 7: Connect Real Data

Tasks:

1. Write adapter for actual STAR / R1SGG data format.
2. Map image paths.
3. Map question / answer / scene graph fields.
4. Run a tiny real-data dry-run.

Stop after Phase 7.

### Phase 8: Real Training

Tasks:

1. Run a small training test.
2. Save checkpoint to `${OUTPUT_ROOT}/checkpoints`.
3. Save logs to `${OUTPUT_ROOT}/logs`.

Stop and report.

---

## 18. Acceptance Criteria

The implementation is acceptable only if:

1. It does not write large files into the Git repository.
2. It respects the real paths:

   * `/root/autodl-tmp/STAR`
   * `/root/autodl-tmp/STAR/r1sgg_data`
   * `/root/autodl-tmp/STAR/STAR`
   * `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs`
3. It activates and uses the `r1sgg` conda environment.
4. It supports dry-run before real training.
5. It has tests for zoom tool, reward, token mask, and GRPO loss.
6. Tool observations are masked out from loss.
7. Only model-generated tokens contribute to GRPO policy loss.
8. Invalid tool actions do not crash training.
9. Invalid bbox does not crash training.
10. Every phase ends with a clear report.

---

## 19. Final Response Format After Each Phase

After each phase, report in this format:

```text
Phase completed:

1. Modified / added files
- ...

2. What was implemented
- ...

3. Commands run
- ...

4. Test results
- ...

5. Data assumptions
- ...

6. Issues or questions
- ...

7. Recommended next phase
- ...
```

Do not continue to the next phase without user confirmation.

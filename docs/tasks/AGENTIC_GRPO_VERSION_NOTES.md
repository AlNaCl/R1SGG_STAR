# Agentic GRPO Version Notes / Agentic GRPO 版本说明

## Snapshot / 快照

- Version label / 版本标签: `agentic-grpo-scaffold-v0`
- Date / 日期: 2026-05-29
- Branch / 分支: `main`
- Scope / 范围: Agentic GRPO/RLVR scaffold for STAR closed-vocabulary scene graph generation.
- 范围说明：面向 STAR 闭集场景图生成的 Agentic GRPO/RLVR 原型脚手架。

## Summary / 摘要

This snapshot adds the first testable Agentic GRPO/RLVR scaffold without starting full model training.

本快照加入第一版可测试的 Agentic GRPO/RLVR 脚手架，但不启动完整大模型训练。

## Included Changes / 包含改动

- Added configurable experiment paths and defaults in `configs/agentic_grpo.yaml`.
- 新增 `configs/agentic_grpo.yaml`，统一实验路径和默认参数。

- Added STAR/R1-SGG RLVR dataset adapters for HuggingFace `load_from_disk` data and JSONL data.
- 新增 STAR/R1-SGG RLVR 数据适配器，支持 HuggingFace `load_from_disk` 数据和 JSONL 数据。

- Added a zoom-in crop tool for large remote-sensing images.
- 新增面向超高分辨率遥感图像的 `zoom_in` 裁剪工具。

- Added JSON action parsing, verifiable reward helpers, token-level loss masking, rollout orchestration, and GRPO loss utilities.
- 新增 JSON action 解析、可验证奖励、token 级 loss mask、rollout 编排和 GRPO loss 工具。

- Added dry-run and train-smoke entry scripts.
- 新增 dry-run 和 train-smoke 启动脚本。

- Updated STAR closed-set inference, baseline visualization, and evaluation helpers.
- 更新 STAR 闭集推理、baseline 可视化和评估辅助脚本。

- Moved task documentation under `docs/tasks/` and restored the project instruction file as `AGENTS.md`.
- 将任务文档整理到 `docs/tasks/`，并将项目指令文件规范为 `AGENTS.md`。

## Validation / 验证

- Dry-run output: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/dry_run_agentic_grpo.json`
- Dry-run 输出：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/dry_run_agentic_grpo.json`

- Train-smoke output: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/train_smoke_agentic_grpo_latest.json`
- Train-smoke 输出：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/train_smoke_agentic_grpo_latest.json`

- Train-smoke checkpoint: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/*/toy_policy.pt`
- Train-smoke 检查点：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/*/toy_policy.pt`

## Output Safety Update / 输出安全更新

- Date / 日期: 2026-05-31
- Dry-run and train-smoke logs/checkpoints now use non-overwriting paths. If the default file or directory already exists, the runner writes a timestamp-suffixed path instead of replacing existing output.
- dry-run 和 train-smoke 的日志/检查点现在使用非覆盖路径。如果默认文件或目录已存在，运行器会写入带时间戳后缀的新路径，而不是替换已有输出。
- `train_smoke_agentic_grpo_latest.json` is created only when that exact path is free; otherwise a suffixed non-overwriting path is reported in the run summary.
- `train_smoke_agentic_grpo_latest.json` 仅在该路径未占用时创建；否则运行摘要会报告一个带后缀的非覆盖路径。

## Model Load Smoke Update / 模型加载冒烟验证更新

- Date / 日期: 2026-05-31
- Added a no-training model-load smoke path for Agentic GRPO. It loads one STAR RLVR sample, converts the Eagle-style prompt to Qwen-VL messages, builds processor inputs, and optionally loads the configured Qwen-VL model without forward/backward or checkpoint writes.
- 新增 Agentic GRPO 的无训练模型加载冒烟路径：读取一条 STAR RLVR 样本，将 Eagle-style prompt 转成 Qwen-VL messages，构造 processor 输入，并可选加载配置的 Qwen-VL 模型；不执行 forward/backward，也不写 checkpoint。
- Verified locally with `/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z`; output log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/model_load_smoke_20260531_101552_419549Z.json`.
- 已使用本地 checkpoint `/root/autodl-tmp/STAR/r1sgg_data/checkpoints/qwen25vl-7b-sft-star-close-20260507_182608Z` 验证；输出日志为 `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/model_load_smoke_20260531_101552_419549Z.json`。

## Generation Smoke Update / 生成冒烟验证更新

- Date / 日期: 2026-05-31
- Added a no-training generation smoke path for one STAR RLVR sample. It loads the configured Qwen-VL model, generates one JSON action, runs at most one `zoom_in`, appends the crop observation, and optionally generates a final answer.
- 新增单样本无训练生成冒烟路径：加载配置的 Qwen-VL 模型，生成一个 JSON action，最多执行一次 `zoom_in`，追加 crop observation，并可选生成 final answer。
- Real checkpoint result: raw dataset prompt mode exposed the current mismatch by reverting to long legacy scene-graph output; `action_only` prompt mode produced valid `zoom_in` and `final_answer` actions.
- 真实 checkpoint 结果：原始数据集 prompt 会回退到很长的旧 scene-graph 输出，暴露当前格式不匹配；`action_only` prompt mode 可生成有效的 `zoom_in` 和 `final_answer` action。
- Successful log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260531_104009_952634Z.json`.
- 成功日志：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260531_104009_952634Z.json`。

## Action-Format SFT Warmup Update / Action 格式 SFT 预热更新

- Date / 日期: 2026-05-31
- Added an action-format SFT dataset builder that exports STAR/RLVR samples as Qwen-style `messages` with `final_answer` JSON actions. The HF `save_to_disk` export stores nested fields as JSON strings for Arrow compatibility.
- 新增 action-format SFT 数据构建器，将 STAR/RLVR 样本导出为 Qwen-style `messages`，目标输出为 `final_answer` JSON action；HF `save_to_disk` 版本会将嵌套字段存为 JSON 字符串以兼容 Arrow。
- Updated `src/sft_sgg.py` to optionally consume prebuilt dataset `messages` through `--use_dataset_messages true`, instead of rebuilding the legacy fenced scene-graph target.
- 更新 `src/sft_sgg.py`，支持通过 `--use_dataset_messages true` 直接使用数据集中的预构建 `messages`，不再强制重建旧 fenced scene-graph target。
- Built a 32-sample short-subgraph warmup dataset: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260531_115430_765141Z/hf_dataset`.
- 已构建 32 条短子图 warmup 数据：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260531_115430_765141Z/hf_dataset`。
- Completed a 1-step LoRA action-format SFT smoke. Output adapter checkpoint: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/sft_action_format_smoke_20260531_120351Z`. Final smoke metrics: loss `0.7815`, grad_norm `0.3373`, mean_token_accuracy `0.8819`.
- 已完成 1-step LoRA action-format SFT smoke。输出 adapter checkpoint：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/sft_action_format_smoke_20260531_120351Z`。最终 smoke 指标：loss `0.7815`，grad_norm `0.3373`，mean_token_accuracy `0.8819`。

## Adapter Generation Smoke Update / Adapter 生成冒烟验证更新

- Date / 日期: 2026-06-01
- Added PEFT adapter loading support to `model_load_smoke` and `generation_smoke` via `--peft-adapter-path`; the shell wrappers also accept `PEFT_ADAPTER_PATH` or `ADAPTER_PATH`.
- 为 `model_load_smoke` 和 `generation_smoke` 增加 PEFT adapter 加载支持，可通过 `--peft-adapter-path` 指定；shell wrapper 同时支持 `PEFT_ADAPTER_PATH` 或 `ADAPTER_PATH` 环境变量。
- Re-ran the requested smoke tests: `pytest tests/test_action_sft_dataset.py tests/test_rlvr_dataset.py tests/test_model_load_smoke.py tests/test_generation_smoke.py -q` -> `12 passed`.
- 已重新运行指定冒烟测试：`pytest tests/test_action_sft_dataset.py tests/test_rlvr_dataset.py tests/test_model_load_smoke.py tests/test_generation_smoke.py -q` -> `12 passed`。
- Compared the base STAR close checkpoint with the 100-step action-format SFT LoRA adapter on `val` sample_index `0` (`star_val_1006`) using `prompt_mode=action_only`.
- 使用 `prompt_mode=action_only` 在 `val` 的 sample_index `0`（`star_val_1006`）上对比 STAR close base checkpoint 和 100-step action-format SFT LoRA adapter。
- Base-only log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260601_083054_979212Z.json`; result: `used_zoom=true`, final `is_valid_json=true`, reward `0.1`, format_reward `1.0`.
- Base-only 日志：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260601_083054_979212Z.json`；结果：`used_zoom=true`，最终 `is_valid_json=true`，reward `0.1`，format_reward `1.0`。
- Adapter log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260601_083004_268869Z.json`; result: `used_zoom=true`, but the final action included extra explanatory text, so final `is_valid_json=false`, reward `0.0`, format_reward `0.0`.
- Adapter 日志：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_smoke_20260601_083004_268869Z.json`；结果：`used_zoom=true`，但 final action 后追加了解释文本，因此最终 `is_valid_json=false`，reward `0.0`，format_reward `0.0`。
- Conclusion: the 100-step action-format SFT adapter did not improve JSON action stability on this smoke sample versus base-only, so the 500-step expansion was not launched.
- 结论：100-step action-format SFT adapter 在该冒烟样本上没有比 base-only 更稳定地输出 JSON action，因此未启动 500-step 扩大训练。
- Added a batch base-vs-adapter generation smoke runner: `python -m src.rl.generation_smoke_batch` and `scripts/generation_smoke_batch_agentic_grpo.sh`. It loads each variant once, runs the same sample indices, writes full per-sample trajectories to a non-overwriting JSON log, and prints an aggregate summary by default.
- 新增 batch base-vs-adapter 生成冒烟脚本：`python -m src.rl.generation_smoke_batch` 和 `scripts/generation_smoke_batch_agentic_grpo.sh`。脚本对每个 variant 只加载一次模型，跑相同 sample indices，将完整逐样本轨迹写入非覆盖 JSON 日志，并默认只在终端打印聚合摘要。
- 20-sample batch log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_batch_smoke_20260601_091412_970598Z.json`. Base metrics: `valid_json_rate=0.40`, `extra_text_rate=0.60`, `final_answer_valid_rate=0.25`, `mean_reward=0.0400`. Adapter metrics: `valid_json_rate=0.30`, `extra_text_rate=0.70`, `final_answer_valid_rate=0.15`, `mean_reward=0.0275`.
- 20 样本 batch 日志：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_batch_smoke_20260601_091412_970598Z.json`。Base 指标：`valid_json_rate=0.40`，`extra_text_rate=0.60`，`final_answer_valid_rate=0.25`，`mean_reward=0.0400`。Adapter 指标：`valid_json_rate=0.30`，`extra_text_rate=0.70`，`final_answer_valid_rate=0.15`，`mean_reward=0.0275`。
- Batch conclusion: the 100-step adapter remains worse than base on action-format stability (`valid_json_rate -0.10`, `extra_text_rate +0.10`), so neither 500-step SFT expansion nor GRPO should be launched from this adapter yet.
- Batch 结论：100-step adapter 在 action-format 稳定性上仍弱于 base（`valid_json_rate -0.10`，`extra_text_rate +0.10`），因此暂不应基于该 adapter 启动 500-step SFT 扩大训练或 GRPO。

## Fixed Format-Only Warmup Update / 修复版格式预热更新

- Date / 日期: 2026-06-02
- Fixed the action-format warmup target mismatch by adding `target_mode=format_only`, switching the default warmup config to `prompt_mode=action_only` + `target_mode=format_only`, and letting `scripts/build_action_sft_dataset.sh` pass `TARGET_MODE` when provided.
- 通过新增 `target_mode=format_only`、将默认 warmup 配置切到 `prompt_mode=action_only` + `target_mode=format_only`、并让 `scripts/build_action_sft_dataset.sh` 支持传入 `TARGET_MODE`，修复 action-format warmup 的提示和目标不一致问题。
- Updated `src/sft_sgg.py` so SFT labels can mask system/user prompt tokens through `--train_on_assistant_only true`; only assistant response tokens contribute to the SFT loss.
- 更新 `src/sft_sgg.py`，支持通过 `--train_on_assistant_only true` mask system/user prompt token；SFT loss 只作用在 assistant response token 上。
- Built a 32-sample fixed format-only warmup dataset: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260602_041337_392056Z/hf_dataset`. First target length was 93 chars with empty `objects` and `relationships`, matching the generation smoke target.
- 已构建 32 条修复版 format-only warmup 数据：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260602_041337_392056Z/hf_dataset`。首条 target 长度为 93 字符，`objects` 和 `relationships` 为空，和 generation smoke 目标一致。
- Completed a fixed 100-step LoRA warmup. Output adapter: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/sft_action_format_fixed_100step_20260602_121200Z`. Final train metrics: `train_loss=0.1602`; last logged steps reached `mean_token_accuracy=1.0`.
- 已完成修复版 100-step LoRA warmup。输出 adapter：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/checkpoints/sft_action_format_fixed_100step_20260602_121200Z`。最终训练指标：`train_loss=0.1602`；最后若干步 `mean_token_accuracy=1.0`。
- Re-ran 20-sample base-vs-fixed-adapter generation smoke on `val` indices 0-19. Log: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_batch_smoke.json`. Base metrics: `valid_json_rate=0.40`, `extra_text_rate=0.60`, `final_answer_valid_rate=0.25`, `mean_reward=0.0400`. Fixed adapter metrics: `valid_json_rate=1.00`, `extra_text_rate=0.00`, `final_answer_valid_rate=1.00`, `mean_reward=0.1000`.
- 已在 `val` indices 0-19 上重新运行 20 样本 base-vs-fixed-adapter generation smoke。日志：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/logs/generation_batch_smoke.json`。Base 指标：`valid_json_rate=0.40`，`extra_text_rate=0.60`，`final_answer_valid_rate=0.25`，`mean_reward=0.0400`。修复版 adapter 指标：`valid_json_rate=1.00`，`extra_text_rate=0.00`，`final_answer_valid_rate=1.00`，`mean_reward=0.1000`。
- Conclusion: the format-only warmup now fixes raw JSON action stability, but it intentionally collapses to immediate empty `final_answer` and does not learn `zoom_in` or scene-graph content. Do not launch GRPO from this adapter as a task policy yet; next add mixed tool-use/content targets or a separate action+content warmup before any GRPO pilot.
- 结论：format-only warmup 现在修复了 raw JSON action 稳定性，但它会按目标直接输出空 `final_answer`，尚未学习 `zoom_in` 或 scene graph 内容。因此暂不应把该 adapter 作为任务策略启动 GRPO；下一步应加入 tool-use/content 混合目标，或单独做 action+content warmup，再考虑 GRPO pilot。

## No-GPU Action+Content Warmup Prep / 无 GPU Action+Content 预热准备

- Date / 日期: 2026-06-02
- Added `action_content` prompts plus `zoom_then_scene_graph` and `mixed_scene_graph` action SFT target modes. The new content targets emit compact scene-graph JSON with `id` and pixel-space xyxy `bbox` only.
- 新增 `action_content` prompt，以及 `zoom_then_scene_graph` 和 `mixed_scene_graph` 两种 action SFT 目标模式。新的内容目标输出紧凑 scene graph JSON，仅保留 `id` 和像素坐标 xyxy `bbox`。
- Fixed STAR bbox handling for action SFT export: source object boxes are xywh, so they are normalized to xyxy before final_answer and zoom target generation.
- 修复 action SFT 导出中的 STAR bbox 处理：源 object box 为 xywh，现在在生成 final_answer 和 zoom target 前统一转换为 xyxy。
- Updated SFT assistant-only label masking for multi-turn agentic messages. All assistant action/final-answer spans can contribute to loss, while system/user/image/tool-observation tokens remain masked.
- 更新多轮 agentic messages 的 SFT assistant-only label mask：所有 assistant action/final-answer span 都可参与 loss，system/user/image/tool-observation token 继续被 mask。
- CPU validation built an 8-sample mixed smoke dataset: `/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260602_105640_104356Z/hf_dataset`; it contains `num_zoom_targets=3` and first record action sequence `zoom_in -> final_answer`.
- CPU 验证已构建 8 条 mixed smoke 数据：`/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs/tmp/action_sft_train_20260602_105640_104356Z/hf_dataset`；其中 `num_zoom_targets=3`，首条样本 action 序列为 `zoom_in -> final_answer`。
- This prep does not run model training or generation and does not require GPU.
- 本准备步骤没有运行模型训练或生成，不需要 GPU。

## Known Limitations / 已知限制

- Full Qwen-VL policy integration into the real Agentic GRPO training loop is not implemented in this scaffold snapshot. The current model-load path is a no-training smoke test.
- 本脚手架快照尚未将完整 Qwen-VL policy 接入真实 Agentic GRPO 训练循环；当前模型加载路径只是无训练冒烟验证。

- The train-smoke path uses a tiny toy parameter/checkpoint to verify dataset loading, loss computation, backward, optimizer step, logging, and checkpoint writing.
- train-smoke 路径使用极小 toy 参数和检查点，仅验证数据加载、loss 计算、反向传播、优化器步骤、日志和检查点写入。

- Scene-graph reward currently uses exact normalized triplet matching; future versions should add bbox-aware and SGG metric-aligned rewards.
- 当前场景图奖励使用归一化三元组精确匹配；后续版本应加入 bbox 感知奖励和与 SGG 指标对齐的奖励。

## Reproducibility Notes / 复现说明

Use the existing environment:

使用现有环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r1sgg
```

Use the standard paths:

使用标准路径：

```bash
export DATA_ROOT=/root/autodl-tmp/STAR
export R1SGG_DATA_ROOT=/root/autodl-tmp/STAR/r1sgg_data
export STAR_RAW_ROOT=/root/autodl-tmp/STAR/STAR
export OUTPUT_ROOT=/root/autodl-tmp/R1SGG_Agentic_GRPO_outputs
```


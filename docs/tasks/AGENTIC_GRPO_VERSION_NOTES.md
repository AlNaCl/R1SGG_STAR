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


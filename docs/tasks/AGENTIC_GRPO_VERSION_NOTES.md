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

## Known Limitations / 已知限制

- Full Qwen-VL policy loading and real Agentic GRPO training are not implemented in this scaffold snapshot.
- 本脚手架快照尚未实现完整 Qwen-VL policy 加载和真实 Agentic GRPO 训练。

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


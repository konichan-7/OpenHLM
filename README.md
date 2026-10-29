# OpenHLM · G1 + Dex3 / π0.5

本仓库保留 G1 + 双 Dex3 的数据处理、π0.5 训练、评估、SIMPLE 仿真部署和
Rerun 回放。状态协议为 **32 维**，动作协议为 **36 维**，单个第一视角 RGB。
支持桌面抓取单任务和同协议的非移动多任务训练。

## 目录

```text
OpenHLM/
├── src/openpi4OpenHLM/
│   ├── examples/simple_g1_dex3/   # 数据审计、划分、多任务清单、评估、部署及测试
│   ├── scripts/                  # 归一化统计、训练、WebSocket 服务
│   ├── src/openpi/                # 模型、数据加载、协议及共享训练基础代码
│   └── packages/openpi-client/    # WebSocket 客户端
├── tools/rerun_g1_dataset/        # 数据集机器人/图像回放
├── assets/g1/                     # 回放所需 URDF、网格和原始许可说明
└── docs/                         # 训练代码导读、方案和已有闭环评估记录
```

## 从这里开始

- [G1 Dex3 使用说明](src/openpi4OpenHLM/examples/simple_g1_dex3/README.md)：环境、数据、单任务/多任务训练、SIMPLE 客户端和服务。
- [训练代码导读](docs/g1_dex3_training_code_walkthrough.md)：沿数据、模型、loss 和 checkpoint 理解代码。
- [闭环评估记录](docs/g1_dex3_simple_closed_loop_results.md)：已有结果及其验证边界。
- [Rerun 回放](tools/rerun_g1_dataset/README.md)：按 episode 或全数据集查看机器人关节和第一视角图像。
- [早期实施方案](docs/g1_dex3_pi05_simple_pipeline_plan.md)：历史设计和阶段记录，当前命令以使用说明为准。

训练与策略服务的工作目录为 `src/openpi4OpenHLM`。SIMPLE/Isaac 仿真在独立
环境中运行，客户端源码保存在 `examples/simple_g1_dex3/simple_client/`，通过
WebSocket 与训练得到的策略通信。SIMPLE 和它的 AMO 控制器由外部 SIMPLE
项目提供，不在本仓库中。

本次精简移除了 HuMI 人体采集、GR00T/SONIC 全身控制、VR 遥操作、原始
34 维双夹爪 G1 路线及 ALOHA/DROID/LIBERO/UR5 示例。共享模型基础代码保留，
以支持 π0.5 的训练、权重加载、归一化和服务。历史版本可从 Git 历史恢复。

## 回放资源

机器人网格使用 Git LFS。新克隆或只有指针文件时，从仓库根目录下载：

```bash
git lfs pull --include="assets/g1/meshes/**"
```

## 来源与许可

本项目从 OpenHLM 完整源码精简，并基于 OpenPI 进行 G1 Dex3 适配。保留
[Apache 2.0 许可](LICENSE)、[第三方声明](NOTICE)、
[OpenPI 许可](src/openpi4OpenHLM/LICENSE)、
[Gemma 条款](src/openpi4OpenHLM/LICENSE_GEMMA.txt) 和
[机器人资源来源](assets/g1/SOURCE.md)。数据、权重、日志和回放生成物不提交到仓库。

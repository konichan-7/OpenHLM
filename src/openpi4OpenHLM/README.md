# G1 + Dex3 π0.5 训练与服务

这是从 OpenHLM/OpenPI 精简出的 G1 + 双 Dex3 训练模块。保留单视角 RGB、
32 维状态、36 维动作的数据协议，支持单任务和固定基座多任务。

## 入口

- [完整使用说明](examples/simple_g1_dex3/README.md)：安装、数据准备、训练、评估和 SIMPLE 部署。
- [训练代码导读](../../docs/g1_dex3_training_code_walkthrough.md)：协议、模型、loss、checkpoint 和多任务命令。
- [归一化统计](docs/norm_stats.md)：统计只使用训练 split。
- [远程推理](docs/remote_inference.md)：WebSocket 服务和独立 SIMPLE 客户端。
- [Rerun 数据回放](../../tools/rerun_g1_dataset/README.md)：机器人关节与第一视角可视化。

## 保留的配置

| 配置 | 用途 |
|---|---|
| `pi05_simple_g1_dex3_tabletop` | 单任务桌面抓取，全参数微调 |
| `pi05_simple_g1_dex3_multitask` | 同协议的非移动多任务训练 |
| `debug` / `debug_restore` / `debug_pi05` | 共享模型与数据加载器的假数据测试 |

共享的 `models/`、`models_pytorch/`、`shared/`、`transforms.py`、训练和权重转换
工具保留。它们承载 π0.5 的模型结构、tokenizer、checkpoint、归一化和推理能力。

## 常用命令

在本目录、已配置好的 Linux/CUDA 环境中执行：

```bash
bash examples/simple_g1_dex3/deploy_server.sh env
bash examples/simple_g1_dex3/deploy_server.sh weights
bash examples/simple_g1_dex3/deploy_server.sh data
bash examples/simple_g1_dex3/deploy_server.sh stats
bash examples/simple_g1_dex3/deploy_server.sh check
```

这些命令用于创建环境、下载数据/基座权重并验证链路，具体前提及训练命令见
[使用说明](examples/simple_g1_dex3/README.md)。多任务通过同一数据配置的
`dataset_manifest` 入口加载；需要先生成多任务清单并单独计算统计。

[Apache 2.0](LICENSE) · [Gemma 条款](LICENSE_GEMMA.txt) · [仓库声明](../../NOTICE)

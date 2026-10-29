# SIMPLE G1 + 双 Dex3-1 抓取：π0.5 全参微调

配套方案：[`docs/g1_dex3_pi05_simple_pipeline_plan.md`](../../../../docs/g1_dex3_pi05_simple_pipeline_plan.md)

协议（写死在第一阶段）：**单视角 + 32 维状态 + 36 维动作**。不是 OpenHLM 原来的 34 维 G1 协议，也不是 43 维全身关节。

| 内容 | 位置 |
|---|---|
| 协议常量 / transforms | `src/openpi/policies/simple_g1_dex3_policy.py` |
| 数据配置 | `LeRobotSimpleG1Dex3DataConfig`（`src/openpi/training/config.py`） |
| 训练配置 | `pi05_simple_g1_dex3_tabletop` |
| episode 过滤 | `src/openpi/training/data_loader.py`（`EpisodeSubsetDataset`） |
| split manifest | `src/openpi/training/manifests.py` |
| 数据审计 / 切分 / 协议测试 / 开环评估 | 本目录 |
| SIMPLE 客户端 | `SIMPLE/src/simple/baselines/pi05_openhlm.py` |

---

## 0. 协议一览

**状态 32 维**（训练数据里的 `states` 列，客户端必须按同一顺序重建）

| 范围 | 内容 |
|---|---|
| 0:3 / 3:5 / 5:7 | 左手 thumb0/1/2、middle0/1、index0/1 |
| 7:14 | 右手 thumb0/1/2、index0/1、middle0/1 |
| 14:21 / 21:28 | 左臂 / 右臂 7 关节（注意：手在前、臂在后） |
| 28:31 | **上一条高层指令**的 torso roll, pitch, yaw |
| 31 | **上一条指令**的高度（绝对，0.75 m 名义站高） |

**动作 36 维**

| 范围 | 内容 | 备注 |
|---|---|---|
| 0:7 / 7:14 | 左 / 右 Dex3，data order | 客户端只对左手做 `(0,1,2,5,6,3,4)` 置换 |
| 14:21 / 21:28 | 左 / 右臂 | 绝对关节目标 |
| 28:31 | torso rpy | 同时给 `waist_qpos` 和 AMO command |
| 31 | 高度（绝对，m） | 下发 AMO 时减 0.75 |
| 32 / 33 | vx / vy | 本任务恒 0 |
| 34 | turning flag | 本任务恒 0，不是角速度 |
| 35 | target yaw | 本任务恒 0 |

已核验的发布包事实（`G1WholebodyTabletopGraspMP-v0`，LeRobot v2.1）：

- 100 episodes / 12,226 帧 / 50 fps / 640×360 AV1 / 单任务
- `states` 32 维、`action` 36 维
- **32:36 四维全为 0，31 维高度恒 0.75**；28:31 的 torso rpy 会在十几毫弧内漂移，**不是常量**

因此 `pin_constant_dims()` 在反归一化之后把 `height=0.75` 与 32:36 四个 0 写回；量化归一化对零 IQR 维度是 `(x-0)/(1+1e-6)`，不写回会把数值噪声喂给 AMO。torso rpy 保持模型输出。

---

## 1. 环境

```bash
cd OpenHLM/src/openpi4OpenHLM
uv sync --no-install-package evdev --no-install-package pynput
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

`evdev`/`pynput` 是 LeRobot 的遥操作依赖，训练和服务都用不到，但它们在无 `Python.h`
的机器上会编译失败；跳过它们不影响本任务。

国内机器建议先设镜像（直连 `files.pythonhosted.org` 往往超时）：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=120
```

或者直接用一键脚本：

```bash
bash examples/simple_g1_dex3/deploy_server.sh env
```

两套环境不要混装：训练/服务用本仓库；SIMPLE（Isaac Sim 4.5 + torch 2.7）用另一套 Python 3.10 环境，只通过 WebSocket 通信。

数据放到 LeRobot 约定的位置：

```
$HF_LEROBOT_HOME/
└── local/g1_dex3_tabletop/            # repo_id
    ├── meta/…
    ├── data/…
    └── videos/…
```

```bash
export HF_LEROBOT_HOME=/data/psi_simple_tabletop/lerobot
unzip G1WholebodyTabletopGraspMP-v0.zip -d /data/psi_simple_tabletop/
mv /data/psi_simple_tabletop/G1WholebodyTabletopGraspMP-v0 \
   $HF_LEROBOT_HOME/local/g1_dex3_tabletop
```

---

## 2. 一键部署

服务器上（A100，已装 uv）：

```bash
cd openpi4OpenHLM
bash examples/simple_g1_dex3/deploy_server.sh env     # 建环境
bash examples/simple_g1_dex3/deploy_server.sh weights # 下载 pi0.5 base PyTorch 权重
bash examples/simple_g1_dex3/deploy_server.sh data    # 下载/解包/审计/切分
bash examples/simple_g1_dex3/deploy_server.sh stats   # 只用 train split 算归一化
bash examples/simple_g1_dex3/deploy_server.sh check   # 协议 + 端到端数据链路
bash examples/simple_g1_dex3/deploy_server.sh smoke   # 100 步训练冒烟
```

`data` 会自动下载 `simple/` 和 `simple-eval/` 两个包。若服务器访问 HuggingFace
超时，在本机下好 `G1WholebodyTabletopGraspMP-v0.zip` 后 `scp` 到
`/data/psi_simple_tabletop/train.zip`，脚本检测到文件已存在就会跳过下载。

`weights` 从 ModelScope 镜像下载 `lerobot/pi05_base` 的 14.5 GB safetensors 并校验
SHA256；训练集群直连 `storage.googleapis.com` 或 `huggingface.co` 都不可靠，这个
镜像实测约 30 MB/s。

已在 A100-SXM4-80GB 上实际跑通：

- `test_protocol.py` 9/9、`test_pipeline_e2e.py`、`test_serve.py`、`test_simple_client.py`
- 100 步全参训练：loss 0.76 → 0.58，峰值 allocated 34.5 GB，稳定约 1.5 step/s
- 3,000 步全参训练：loss 1.138 → 0.024，训练内 eval MSE 0.682 → 0.009
- 验证集 1,500/2,000/3,000 步：右手 MAE `0.0303 → 0.0229 → 0.0174`，右臂 `0.0453 → 0.0424 → 0.0327`
- `run1/3000` checkpoint 直接推理 `(30,36)`，热推理约 404 ms/chunk
- 真实 WebSocket 服务 + 独立客户端请求通过，metadata 和常量维写回均正确

## 3. 开工前检查（不占 GPU）

```bash
# 协议单测：维度、手部置换、常量维度、transforms
PYTHONPATH=src python examples/simple_g1_dex3/test_protocol.py

# 带 SIMPLE 源码交叉验证手部映射（推荐）
SIMPLE_ROOT=/path/to/SIMPLE PYTHONPATH=src python examples/simple_g1_dex3/test_protocol.py

# 数据审计：维度、NaN、常量维、episode 长度、视频帧数
PYTHONPATH=src python examples/simple_g1_dex3/audit_data.py \
  --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop --check-videos
```

端到端数据链路（需要已装好的真实环境，不需要 GPU）：

```bash
HF_LEROBOT_HOME=$HF_LEROBOT_HOME PYTHONPATH=src python examples/simple_g1_dex3/test_pipeline_e2e.py \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

它会真实跑 LeRobotDataset → episode 过滤 → repack → transforms → 量化归一化 →
模型输入 spec，并检查归一化往返误差 < 1e-3。

`audit_data.py` 若报 `frozen dims are [...], protocol pins [28..35]`，说明发布包换过版本，**先改 `CONST_*` / `pin_constant_dims()` 再训练**。

---

## 4. 冻结 split

发布包 100 个 episode 全部标记为 `train`，没有官方留出集。按 `environment_config` 分组切分，避免近似场景泄漏：

```bash
python examples/simple_g1_dex3/make_split.py \
  --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop \
  --out assets/simple_g1_dex3/tabletop_split.json \
  --val-fraction 0.2 --seed 0
```

生成的 manifest 带 `sha256`，`manifests.load()` 会校验；训练、统计、评估都读同一份。

---

## 5. 归一化统计（只用 train split）

```bash
export OPENPI_DATA_HOME=$PWD
PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_tabletop \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

产物：`assets/pi05_simple_g1_dex3_tabletop/local/g1_dex3_tabletop/norm_stats.json`（`state` 32 维、`actions` 36 维）。

`--episode-manifest` 会同时作用于训练和统计；**不要**用全量数据算统计再训练子集。

---

## 6. 训练（A100 80GB）

先冒烟 50–100 步，确认 loss/梯度有限、显存可控：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py \
  pi05_simple_g1_dex3_tabletop \
  --exp_name smoke \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --num_train_steps 100 \
  --save_interval 50 \
  --log_interval 10 \
  --wandb_enabled False
```

正式第一轮（默认 3,000 步，`save_interval=500`）：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py \
  pi05_simple_g1_dex3_tabletop \
  --exp_name run1 \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

显存不够时依次尝试：`--batch_size 2` → `--enable_gradient_checkpointing True`（默认已开）→ 两卡 DDP。**不要**为了塞进去去改 `action_dim` 或 horizon。

关键配置（`config.py` 中可覆盖）：

| 项 | 值 |
|---|---|
| `action_dim` / `action_horizon` | 36 / 30 |
| 精度 | bf16 |
| batch size | 4（单卡起点） |
| 峰值学习率 | 3e-5，warmup 300 |
| EMA / LoRA / 冻结 | 全关，全参数微调 |
| 动作投影 | `weight_surgery`：前 32 列用预训练权重，后 4 列新初始化 |

---

## 7. 开环评估

```bash
PYTHONPATH=src python examples/simple_g1_dex3/eval_openloop.py \
  --checkpoint checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000 \
  --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --episodes 80 81 82 --stride 5 \
  --out /tmp/openloop.json
```

输出按块分列 MAE（左手 / 右手 / 左臂 / 右臂 / torso rpy / 高度 / 底座指令）。`base_cmd` 的 MAE 应接近 0（常量维），否则说明反归一化或统计有问题；`torso_rpy` 允许有小的正常误差。

只想验证 checkpoint 本身时：

```bash
python examples/simple_g1_dex3/test_checkpoint.py \
  --checkpoint checkpoints/pi05_simple_g1_dex3_tabletop/run1/5000 \
  --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop \
  --episode 1 --frame 30
```

服务起来后，用独立进程验证网络契约：

```bash
python examples/simple_g1_dex3/test_remote_client.py \
  --host 127.0.0.1 --port 8000 \
  --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop
```

---

## 8. 起服务 + SIMPLE 闭环

服务端（A100，与训练同环境）：

```bash
PYTHONPATH=src python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.dir checkpoints/pi05_simple_g1_dex3_tabletop/run1/5000 \
  --policy.config pi05_simple_g1_dex3_tabletop
```

客户端（SIMPLE 环境，先安装本目录保留的适配器）：

```bash
bash /path/to/OpenHLM/src/openpi4OpenHLM/examples/simple_g1_dex3/install_simple_client.sh /path/to/SIMPLE
cd /path/to/SIMPLE
python -m simple.cli.eval --policy pi05_openhlm --env simple/G1WholebodyTabletopGraspMP-v0 \
  --host <server-ip> --port 8000
```

`--policy pi05_openhlm` 会解析到 `Pi05OpenhlmAgent`（`snake_to_pascal` 规则）。客户端会在握手时校验 `action_protocol == "simple_g1_dex3"` 与 `action_dim == 36`，对不上直接报错而不是跑出一个静默错的 rollout。

客户端相对原 `pi05.py` 的改动：

1. 发送完整 32 维状态（原 baseline 发 `states[0,:28]`，丢掉历史 rpy/高度）；
2. 不再把输出截到 28 维；
3. 检查 `(H,36)`、有限值、NaN；
4. `reset()` 同时清队列、历史命令、步数；
5. 历史命令初值用 `[0, -0.15, 0, 0.75]`，与数据一致（原 baseline 用 `[0,0,0,0.75]`）。

---

## 9. M2 前置：无 VLA 回放

在跑第一条 VLA rollout 之前，先用原始 episode 的动作回放，确认 AMO 权重、场景资产、Dex3 接触、PD 配置都对得上：

```bash
python -m simple.cli.eval --policy replay --env simple/G1WholebodyTabletopGraspMP-v0 \
  --episode <idx>
```

回放失败时不要加训练步数，先查资产版本 / reset / 时间轴。

---

## 10. 已知边界

- 只做右手抓取抬升（`reward >= 0.9`，约 1.8 cm 抬升），不是放置，也不是双手任务。
- 训练场景闭环只算拟合检查；泛化结论只能用官方评测包。
- 底座指令（32:36）在本任务中是常量，模型在这四个维度上没有可学习信号，由 `pin_constant_dims()` 兜底；torso rpy 仍有小幅真实变化，交给模型。
- 8 GB 笔记本只做数据检查和客户端调试；训练与推理都在 A100 上。


## 11. 非移动多任务训练

使用 `pi05_simple_g1_dex3_multitask` 配置，保留同一个 32-state / 36-action 协议。
通过 `make_multitask_manifest.py` 选择各任务的非移动 episodes，再分别计算统计和
启动训练。具体命令见 [训练代码导读第 21 节](../../../../docs/g1_dex3_training_code_walkthrough.md#21-多任务训练命令)。

对应的链路检查为 `test_multitask_pipeline.py`。当前统计覆盖各 repo 的 train split；
多任务推理的 prompt 需要客户端显式传入。`deploy_server.sh` 仍是桌面抓取单任务入口。

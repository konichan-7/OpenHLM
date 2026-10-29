# G1 + 双 Dex3-1 训练代码详解

本文只讲解你为 **Unitree G1 + 左右 Dex3-1** 新增和修改的训练代码，不展开其它机器人的配置。

代码讲解基线：`main@019d766`。2026-10-29 精简后，本篇涉及的 G1 数据/训练协议保留，
无关机器人适配器和 RLDS 分支已移除；源码行号可能随精简移动。相关实现主要来自以下提交：

| 提交 | 内容 |
|---|---|
| `d289fc6` | 32 维状态 / 36 维动作协议、split-aware 数据管线、单任务配置 |
| `401a717` | 数据审计、split、训练链路测试、开环评估、SIMPLE 客户端 |
| `21cbeb0` | PyTorch 依赖、`~` 路径展开、推理编译时机修正 |
| `cff340c` | 客户端延迟连接，避免 60 步热身后的 WebSocket 空闲超时 |
| `0bcb52b` | 固定基座多任务 manifest、多 repo 数据加载、三任务配置 |

此外，[g1_dex3_pi05_simple_pipeline_plan.md](g1_dex3_pi05_simple_pipeline_plan.md)
记录了设计约束、验收标准和 A100 实测结果，
[g1_dex3_simple_closed_loop_results.md](g1_dex3_simple_closed_loop_results.md)
记录了 checkpoint `run1/3000` 在 SIMPLE 中的闭环结果。

---

## 1. 这套代码解决什么问题

原始 OpenHLM G1 协议是 34 维：

```text
左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1
+ 左腿 6 + 右腿 6 + 腰 3 + root 3
= 34
```

你的 G1 + 双 Dex3-1 任务使用的是 PSI/SIMPLE 数据协议，不是上述 34 维协议：

```text
输入 state: 32 维
输出 action: 36 维
视觉输入:   1 路第一视角
任务:      固定基座桌面抓取 / 非移动操作
动作块:     30 步，50 fps，对应 0.6 秒
模型:       π₀.₅ PyTorch 全参数微调
```

因此，这套代码的核心不是“把一个 34 维配置改成 36”，而是建立一条完整、可验证的数据与控制契约：

```mermaid
flowchart LR
    A[SIMPLE LeRobot 数据] --> B[audit_data.py]
    B --> C[make_split.py / make_multitask_manifest.py]
    C --> D[EpisodeSubsetDataset / ConcatDataset]
    D --> E[Repack]
    E --> F[SimpleG1Dex3Inputs]
    F --> G[量化归一化]
    G --> H[Resize + π0.5 文本/状态 token 化 + Pad]
    H --> I[π0.5 PyTorch 训练]
    I --> J[36 维入口/出口投影 weight surgery]
    J --> K[checkpoint + norm stats + metadata]
    K --> L[OpenHLM WebSocket policy server]
    L --> M[Pi05OpenhlmAgent]
    M --> N[G1 关节 + Dex3 PD + AMO]
```

---

## 2. 代码地图

你新增或主要修改的代码可以按职责分成六层。

### 2.1 协议层

[simple_g1_dex3_policy.py](../src/openpi4OpenHLM/src/openpi/policies/simple_g1_dex3_policy.py)

负责：

- 定义 `STATE_DIM=32`、`ACTION_DIM=36`；
- 训练时将 LeRobot 数据映射为 π₀.₅ `Observation`；
- 检查 state/action 维度；
- 为缺失的左右腕相机创建 masked placeholder；
- 推理输出截断为 36 维；
- 将五个任务常量维写回固定值；
- 导出动作分块工具，供评估和测试使用。

### 2.2 数据配置层

[config.py](../src/openpi4OpenHLM/src/openpi/training/config.py)

新增：

- `LeRobotSimpleG1Dex3DataConfig`；
- `pi05_simple_g1_dex3_tabletop`；
- `pi05_simple_g1_dex3_multitask`；
- 单任务 episode manifest 入口；
- 多任务 `dataset_manifest` 入口；
- 36 维模型的协议 metadata。

### 2.3 数据加载层

[data_loader.py](../src/openpi4OpenHLM/src/openpi/training/data_loader.py)

新增：

- `EpisodeSubsetDataset`：从全量数据中只选择固定 episode；
- `episode_frame_indices()`：从 `episode_index` 列构造扁平帧索引；
- `ConcatDataset`：拼接多个任务的 LeRobot repo；
- `create_torch_dataset()` 的 per-repo 加载函数；
- 多 repo fps 一致性检查；
- `action_sequence_keys=("action",)` 对应的 chunk 生成。

### 2.4 Split 与数据审计

新增：

- [audit_data.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/audit_data.py)
- [make_split.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/make_split.py)
- [make_multitask_manifest.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/make_multitask_manifest.py)
- [manifests.py](../src/openpi4OpenHLM/src/openpi/training/manifests.py)

这些文件保证训练开始前就能发现：

- state/action 维度不对；
- 某 episode 有 NaN/Inf；
- parquet 与视频帧数不一致；
- train/val 有 episode 重叠；
- split 改变了 episode 集合；
- 多任务数据中混入了移动 episode；
- 不同任务 fps 不一致。

### 2.5 训练主循环层

[train_pytorch.py](../src/openpi4OpenHLM/scripts/train_pytorch.py)

大部分主循环来自 OpenPI；与你任务直接相关的修改有：

- PyTorch 路径支持 `~/.cache/...`；
- `action_dim > 32` 时执行 32→36 维 action projection weight surgery；
- 训练开始时不再对推理采样路径执行 `max-autotune`；
- 可通过 `--episode-manifest` 应用单任务 train split；
- checkpoint 中保存模型、metadata 和训练时 norm stats。

### 2.6 评估、部署和客户端层

新增：

- [test_protocol.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/test_protocol.py)
- [test_pipeline_e2e.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/test_pipeline_e2e.py)
- [test_multitask_pipeline.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/test_multitask_pipeline.py)
- [eval_openloop.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/eval_openloop.py)
- [pi05_openhlm.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/simple_client/pi05_openhlm.py)
- [deploy_server.sh](../src/openpi4OpenHLM/examples/simple_g1_dex3/deploy_server.sh)

---

## 3. 固定协议：32 维状态与 36 维动作

协议是整个代码中最重要的不变量。模型参数可以换，horizon 可以调，但训练、服务端和客户端对每一维
含义的解释必须一致。

### 3.1 State 32 维

```text
索引       内容
0:3        左手 Dex3: thumb 0/1/2
3:5        左手 Dex3: middle 0/1
5:7        左手 Dex3: index 0/1
7:10       右手 Dex3: thumb 0/1/2
10:12      右手 Dex3: index 0/1
12:14      右手 Dex3: middle 0/1
14:21      左臂: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
21:28      右臂: 同左臂
28:31      上一条已执行指令的 torso roll/pitch/yaw
31         上一条已执行指令的绝对高度，单位 m
```

关键点：

1. **手在臂前**。
2. 左右手的 finger 子顺序不同：
   - 左手数据顺序是 `thumb / middle / index`；
   - 右手数据顺序是 `thumb / index / middle`。
3. 最后 4 维不是当前机器人躯干状态，而是**上一条高层指令**。
4. 这个 state 只包含上肢、双手和 torso 指令历史，不包含腿和 root 速度。

### 3.2 Action 36 维

```text
索引       内容
0:7        左手 Dex3，数据顺序 thumb/middle/index
7:14       右手 Dex3，数据顺序 thumb/index/middle
14:21      左臂 7 关节
21:28      右臂 7 关节
28:31      torso roll/pitch/yaw
31         绝对高度 m
32         vx
33         vy
34         turning flag
35         target yaw
```

动作也是**手在臂前**，和原始 OpenHLM 的“臂在前后接夹爪”不同。

### 3.3 哪些维度是常量

在发布包的 12,226 帧上核验：

```text
action[31]    = 0.75
action[32:36] = 0
```

即高度固定为名义站高 0.75 m，平移、转向和 target yaw 都是 0。

但 `action[28:31]` 的 torso rpy **不是常量**，会轻微漂移，因此仍交给模型预测。

常量维存在一个数值问题：量化归一化分母约为 `1e-6`，反归一化后模型输出可能被放大成噪声。
因此代码在两个位置写回常量：

- 服务端输出 transform：`pin_constant_dims()`；
- SIMPLE 客户端：再次用常量覆盖。

这样可以保证即使某一层漏掉处理，也不会把数值噪声直接发给 AMO。

### 3.4 Hand permutation

数据集顺序到 Dex3 PD 顺序的映射是：

```python
HAND_TO_PD_ORDER = (0, 1, 2, 5, 6, 3, 4)
```

它把：

```text
[thumb0, thumb1, thumb2, middle0, middle1, index0, index1]
```

变为：

```text
[thumb0, thumb1, thumb2, index0, index1, middle0, middle1]
```

这个置换只对左手使用；右手数据本身已经符合 PD 顺序。

这里不要对模型输出调用 `from_psi0_upper_joints()`。该 helper 会把左手再置换一次，最终造成
索引手指和中指交换两次后仍然错误的静默控制错误。

---

## 4. 训练数据从哪来

### 4.1 数据集事实

当前单任务数据来自 SIMPLE：

```text
G1WholebodyTabletopGraspMP-v0
LeRobot v2.1
100 episodes
12,226 frames
50 fps
640×360 第一视角视频
states: 32
action: 36
```

LeRobot 目录预期为：

```text
$HF_LEROBOT_HOME/local/g1_dex3_tabletop/
├── meta/
├── data/
└── videos/
```

训练配置中的 `repo_id` 是 `local/g1_dex3_tabletop`。

### 4.2 数据审计

`audit_data.py` 不导入 openpi，直接读 parquet 和 metadata，主要检查：

- `meta/info.json` 中有 `states`、`action` 和至少一个 `observation.images.*`；
- 每个 parquet 有 `states/action/episode_index/frame_index`；
- state 是 32 维，action 是 36 维；
- 无 NaN/Inf；
- row 数与 `meta/episodes.jsonl` 中的 `length` 一致；
- episode 没有空数据；
- 每个 `episode_index` 都存在；
- 可选解码视频并检查视频帧数与 parquet row 数一致；
- 打印每维 min/max/std；
- 检查常量 action 维是否是约定的 `31:36`。

运行：

```bash
cd src/openpi4OpenHLM
PYTHONPATH=src python examples/simple_g1_dex3/audit_data.py \
  --dataset-root "$HF_LEROBOT_HOME/local/g1_dex3_tabletop" \
  --check-videos
```

如果审计报 frozen dims 与 protocol pins 不一致，说明数据协议已经变了，不能直接沿用当前训练配置。

---

## 5. 单任务 train/val split

### 5.1 为什么不能直接随机拆 episode

SIMPLE 数据中不同 episode 可能来自几乎相同的 `environment_config`。如果同一场景组的 episode
同时进入 train 和 val，验证集就不是真正的留出集。

`make_split.py` 先按 `environment_config` 的规范化哈希分组，再按组划入 val：

```text
episode_index
→ environment_config JSON
→ 删除可选 ignore keys
→ sort_keys JSON
→ sha256
→ group key
→ 打乱 group key
→ 整组加入 val，直到约 20%
```

它不会把同一场景组拆到两边。

### 5.2 manifest 内容

生成：

```json
{
  "repo_id": "local/g1_dex3_tabletop",
  "train": [0, 1, 2],
  "val": [80, 81, 82],
  "frames_train": 9794,
  "frames_val": 2432,
  "groups": 100,
  "seed": 0,
  "val_fraction": 0.2,
  "sha256": "..."
}
```

`manifests.load()` 会拒绝：

- 缺少 train 或 val；
- train/val 有交集；
- train 为空；
- 保存的 sha256 与 train/val 内容不一致。

训练和统计只需要 train：

```bash
PYTHONPATH=src python examples/simple_g1_dex3/make_split.py \
  --dataset-root "$HF_LEROBOT_HOME/local/g1_dex3_tabletop" \
  --out assets/simple_g1_dex3/tabletop_split.json \
  --val-fraction 0.2 --seed 0
```

### 5.3 Split 如何进入训练

命令行：

```text
--episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

处理链：

```text
TrainConfig.episode_manifest
→ config.resolve_manifest()
→ manifests.load()
→ data.episode_indices = tuple(train)
→ create_torch_dataset()
→ EpisodeSubsetDataset
```

`EpisodeSubsetDataset` 不改 parquet，只把数据集局部索引映射到原数据帧：

```text
subset[0] -> full_dataset[frame_indices[0]]
subset[1] -> full_dataset[frame_indices[1]]
```

因此底层 LeRobot 仍能基于原始帧生成 `[30, 36]` action chunk，并保证 chunk 不跨越 episode。

---

## 6. 多任务 manifest

### 6.1 目的

多任务配置加入三个固定基座任务：

```text
tabletop grasp
bend-pick MP
bend-pick teleop
```

训练时不希望混入真正移动的 episode，因此 `make_multitask_manifest.py` 按动作中的高层底座命令过滤：

```text
max |action[:, 32]| = max |vx|
max |action[:, 33]| = max |vy|
max |action[:, 34]| = max |turning|
```

默认阈值：

```text
|vx|      <= 0.02
|vy|      <= 0.02
|turning| <= 0.5
```

三条件同时满足才保留。

这不是完全没有躯干运动：高度和 torso rpy 仍可变化，被过滤的是水平移动/转向命令。

### 6.2 多任务的 split

每个 repo 分别：

1. 审计 episode；
2. 过滤 loco episode；
3. 按 `environment_config` 分组；
4. 组级别切 train/val；
5. 保存每个 repo 的 train/val episode 与帧数；
6. 记录被过滤 episode 的各项最大值。

最终 manifest 结构：

```json
{
  "selection": {
    "max_abs_vx": 0.02,
    "max_abs_vy": 0.02,
    "max_abs_turning": 0.5
  },
  "datasets": [
    {
      "repo_id": "local/task-a",
      "train": [0, 2],
      "val": [1],
      "frames_train": 1000,
      "frames_val": 500,
      "skipped_loco_episodes": []
    }
  ],
  "frames_train": 1000,
  "frames_val": 500,
  "sha256": "..."
}
```

该提交记录的三任务集合规模为：

```text
train: 238 episodes / 36,282 frames
val:    59 episodes /  9,076 frames
```

注意 `config.py` 中当前注释写着 “45k frames”，这是旧估算，和提交正文记录的 36,282 帧不一致；
实际训练迭代数应以生成的 manifest 和 `frames_train` 为准。

### 6.3 过滤为什么会漏掉其它 repo

`make_multitask_manifest.py` 的 `--repo` 接收的是 repo_id，实际目录仍由：

```text
HF_LEROBOT_HOME / repo_id
```

决定。若 repo_id 指向的文件不存在，脚本直接报错，不会猜测路径。

---

## 7. DataLoader 的实现

### 7.1 单 repo

`create_torch_dataset()` 的核心流程：

```python
metadata = LeRobotDatasetMetadata(repo)
dataset = LeRobotDataset(
    repo,
    delta_timestamps={
        "action": [t / metadata.fps for t in range(action_horizon)]
    },
)
```

`action_horizon=30`、`fps=50` 时，从第 `i` 帧读取：

```text
action[i], action[i+1], ..., action[i+29]
```

数据集列名是单数 `action`，所以：

```python
action_sequence_keys = ("action",)
```

不能沿用 OpenPI 默认的复数 `actions`。

如果开启 `episode_indices`：

```python
frame_indices = np.flatnonzero(episode_index in selected_episodes)
dataset = EpisodeSubsetDataset(dataset, frame_indices)
```

这一步只过滤帧，不改变底层 action chunk 的时间逻辑。

如果开启 `prompt_from_task`，还会用当前样本的 `task_index` 从该 repo 的 `metadata.tasks`
查语言指令。

### 7.2 多 repo

`dataset_manifest` 存在时：

```text
逐 repo
→ 读取 metadata
→ 检查 fps
→ 读取该 repo 的 train episode 列表
→ EpisodeSubsetDataset
→ PromptFromLeRobotTask
→ ConcatDataset
```

`ConcatDataset` 保存每个子数据集累计结尾，通过 `bisect_right` 把全局索引映射到具体 repo。

例如：

```text
repo A: 1000 帧
repo B: 500 帧
global index 999  -> A[999]
global index 1000 -> B[0]
```

每个子集按自己的 `task_index -> prompt` 映射，所以三个任务不会共用同一条语言指令。

注意：拼接后按帧数自然加权。帧数更多的任务采样概率更高；当前代码没有 `WeightedRandomSampler`
或任务均衡采样。

### 7.3 当前多任务 manifest 的校验边界

`make_multitask_manifest.py` 计算 `sha256`，设计文档也强调可追溯性。
但当前 `data_loader.py` 读取 `dataset_manifest` 时只做 `json.loads()`，并未调用
`manifests.load()` 校验多任务 manifest 的 sha256。

因此：

- 单任务 `episode_manifest` 会校验哈希；
- 多 repo `dataset_manifest` 目前只检查结构和 fps，不校验哈希；
- 若要严格复现多任务训练，需额外把多任务 manifest 的哈希校验补上，或把哈希写入训练 metadata。

---

## 8. 训练 Transform

单任务的数据流：

```text
LeRobot row:
  observation.images.egocentric
  states[32]
  action[H, 36]
  task_index

RepackTransform:
  observation/image   = observation.images.egocentric
  observation/state   = states
  actions             = action
  prompt              = prompt

SimpleG1Dex3Inputs:
  image               -> base_0_rgb
  zero image          -> left_wrist_0_rgb, mask=False
  zero image          -> right_wrist_0_rgb, mask=False
  state               -> state
  actions             -> actions

Normalize:
  state[32]           -> state[32]
  actions[H,36]       -> actions[H,36]

ModelTransformFactory:
  ResizeImages(224,224)
  TokenizePrompt(state, π0.5)
  PadStatesAndActions(36)

Observation:
  image: 3 × [B,224,224,3] -> model 内部 [B,3,224,224]
  mask:  3 × [B]
  state: [B,36]
  prompt:[B,200]
```

### 8.1 `SimpleG1Dex3Inputs`

代码同时接受两种输入键：

- 训练原始键：`observation.images.egocentric`、`states`、`action`；
- 训练 repack 后/服务端使用的键：`observation/image`、`observation/state`、`actions`。

它会：

- 强制 `state.shape[-1] == 32`；
- 强制 `action.shape[-1] == 36`；
- 把浮点图像转成 `uint8 HWC`；
- 把 CHW 图像转成 HWC；
- 给左右腕相机填零并把 mask 设为 false；
- 将 prompt 转成 Python `str`。

缺相机的零图仍会进入视觉塔，但 attention mask 会屏蔽它；它不是免费的“没有计算”。

### 8.2 为什么 state 最后要 pad 到 36

π₀.₅ 把 state 离散化写进文本，因此 `TokenizePrompt` 使用的是归一化后的真实 32 维 state。
随后 `PadStatesAndActions(36)` 在 state 末尾补 4 个零，让 `Observation.state` 的最大维度和 36 维
action 接口一致。

对于 pi0.5 推理，后缀不再把 state 投影为连续 token，因此这 4 个补零不会改变离散状态文本。

### 8.3 为什么不使用 delta action

`use_delta_joint_actions=False`。

原因是 36 维动作混合了：

- 绝对关节目标；
- 绝对高度；
- 高层 base command；
- turning flag。

统一减当前 state 会改变这些量的语义，特别是手、臂、高度和 command token。

---

## 9. 归一化

单任务和多任务均使用量化归一化：

```text
x_norm = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
```

统计来自 `compute_norm_stats.py`：

```text
create_torch_dataset()
→ Repack
→ SimpleG1Dex3Inputs
→ RunningStats.update(state)
→ RunningStats.update(actions)
→ assets/<config>/<asset_id>/norm_stats.json
```

这里刻意不执行 `ResizeImages` / `TokenizePrompt` / `PadStatesAndActions`，因此统计维度是：

```text
state  : 32
actions: [H, 36] 中所有 horizon 位置
```

### 9.1 train-only 统计

单任务命令：

```bash
cd src/openpi4OpenHLM
export HF_LEROBOT_HOME=/data/psi_simple_tabletop/lerobot
PYTHONPATH=src uv run python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_tabletop \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

多任务配置：

```bash
PYTHONPATH=src uv run python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_multitask
```

多任务 config 会把 `dataset_manifest` 中的每个 repo 和 train episode 拼接到一起，统计覆盖三个任务。

### 9.2 常量维处理

`RunningStats.get_statistics()` 会把量化区间小于 0.04 的维度改成 `q01=0, q99=1`。

这比用接近零的区间做分母安全，但也意味着该维在归一化空间中的信息很弱或被压平。任务本身对这些
常量维没有学习需求，因此服务端最终直接写回固定值：

```text
height = 0.75
vx = vy = turning = target_yaw = 0
```

### 9.3 统计与 checkpoint

PyTorch checkpoint 保存：

```text
<step>/assets/<asset_id>/norm_stats.json
```

推理时 `policy_config.create_trained_policy()` 默认从 checkpoint 中读 norm stats，而不是重新从
当前配置文件读。这样可以防止训练后配置或统计数据发生变化。

---

## 10. 训练配置

### 10.1 单任务：`pi05_simple_g1_dex3_tabletop`

```text
模型:          π₀.₅
action_dim:    36
action_horizon:30
discrete_state:True
数据:          local/g1_dex3_tabletop
视觉:          1 路
量化归一化:    True
batch_size:    4
num_workers:   4
训练精度:      bf16
AdamW:         beta=(0.9,0.95), wd=1e-10, clip=1.0
学习率:        warmup 300 -> peak 3e-5 -> decay 10k -> final 3e-6
训练步数:      3,000
保存间隔:      500
EMA:           关闭
冻结:          关闭，全参数微调
```

`policy_metadata` 固定为：

```json
{
  "action_protocol": "simple_g1_dex3",
  "action_dim": 36,
  "state_dim": 32,
  "action_horizon": 30,
  "prompt": "grasp the object on the tabletop."
}
```

这个 metadata 会出现在 WebSocket 握手中，客户端会检查它，防止连接到错误的 34 维 OpenHLM policy。

### 10.2 多任务：`pi05_simple_g1_dex3_multitask`

```text
模型与动作协议: 同上
数据:          dataset_manifest 中的多个 repo
asset_id:      g1_dex3_multitask
batch_size:    4
学习率:        warmup 500 -> peak 3e-5 -> decay 12k -> final 3e-6
训练步数:      12,000
保存间隔:      1,000
```

多任务使用 `prompt_from_task=True`，训练样本的语言指令由各自 LeRobot repo 的 task 决定。

但 `policy_metadata["prompt"]` 仍写着 tabletop grasp，它更像“必须提供 prompt 时的 fallback/协议
字段”，不是多任务推理时自动选择任务的机制。闭环部署时必须由客户端传入当前任务对应的 instruction。

---

## 11. 32→36 维动作头适配

π₀.₅ base 的 action 投影按 32 维训练。你的配置是 36 维。

`PI0Pytorch.__init__()` 中：

```python
action_proj_dim = config.action_dim if config.action_dim > 32 else 32
self.action_in_proj = Linear(action_proj_dim, expert_width)
self.action_out_proj = Linear(expert_width, action_proj_dim)
```

所以模型会创建：

```text
action_in_proj : [hidden, 36]
action_out_proj: [36, hidden]
```

但 base checkpoint 只有 32 列/行，不能直接 `load_state_dict(strict=True)`。

### 11.1 默认 `weight_surgery`

[train_pytorch.py](../src/openpi4OpenHLM/scripts/train_pytorch.py)
`load_model_with_action_proj_expansion()`：

```text
action_in_proj.weight[:, :32]  <- base 权重
action_in_proj.weight[:, 32:]  <- Xavier 初始化

action_out_proj.weight[:32, :] <- base 权重
action_out_proj.weight[32:, :] <- Xavier 初始化

action_out_proj.bias[:32]      <- base bias
action_out_proj.bias[32:]      <- 0
```

这样做保留了 base 模型前 32 维动作表示，只让新增的 4 维（vx、vy、turning flag、target yaw）
从头学习。

另一种 `random` 模式会丢弃全部 action projection 预训练权重，当前任务没有使用。

### 11.2 预训练权重路径

训练读取：

```text
<pytorch_weight_path>/model.safetensors
```

当前配置：

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

`21cbeb0` 修正了 `os.path.expanduser()`，否则 `~` 会作为普通字符传给文件系统。

### 11.3 为什么训练不再编译 sampling

原 `PI0Pytorch` 构造函数无条件调用：

```python
self.compile_for_inference()
```

这会对推理用的 `sample_actions()` 做 `torch.compile(mode="max-autotune")`，训练根本不用，
却要承担启动开销。

当前修改：

- 训练构造模型时不编译采样；
- `policy_config.create_trained_policy()` 加载 checkpoint 后才调用
  `compile_for_inference()`；
- 训练可继续用 `--enable-training-compile` 单独编译 `forward()`。

---

## 12. G1 + Dex3 在 π₀.₅ 中的前向

### 12.1 输入侧

```text
第一视角图像 640×360
→ resize_with_pad 到 224×224
→ SigLIP 视觉编码
```

state 32 维先量化归一化，再离散成 256 个 bin，拼进文本：

```text
Task: grasp the object on the tabletop., State: <32 discretized values>;
Action:
```

左右腕视觉占位 token 的 mask 是 false，因此前缀的有效视觉上下文只有第一视角。

### 12.2 动作侧

训练目标 action chunk：

```text
a ∈ R^(30×36)
```

对每一步 flow matching：

```text
t   ∈ [0.001, 1]
ε   ~ N(0, I)
x_t = t * ε + (1 - t) * a
u_t = ε - a
```

36 维 action 经过 `action_in_proj` 变成 30 个连续 action token。π₀.₅ 用 AdaRMS 把时间步注入
action expert。

视觉语言前缀和 action expert 后缀做联合 attention。前缀不能被动作后缀修改；动作 suffix 可以读取
视觉、语言和离散状态。

### 12.3 输出侧

模型输出 shape：

```text
[B, 30, 36]
```

训练损失：

```text
L = mean((v_pred - (noise - action))²)
```

这里 loss 是归一化动作空间中的 velocity MSE，不是关节角 MAE，也不是任务成功率。

---

## 13. 单个训练 step

主循环来自 `scripts/train_pytorch.py`。

```text
1. DataLoader 给出 Observation + actions[H,36]
2. image/state/prompt/action 移到 GPU
3. 按 global_step 设置 AdamW 当前 LR
4. 模型内部采样 noise 和 time
5. 构造 noisy action x_t
6. 视觉 token + 离散状态/语言 token + 36 维 action token 前向
7. 预测 velocity
8. MSE
9. backward
10. global grad norm clip
11. AdamW step
12. 更新 global_step
13. 每 500/1000 步保存 checkpoint
```

当前配置是全文全参微调：

```text
视觉塔、语言主干、action expert、action projections 都参与优化
```

PyTorch 训练器没有把 `freeze_filter` 转成 `requires_grad=False`；这里的
`freeze_filter=nnx.Nothing` 与“全参训练”一致，但不要误认为改了这个字段就能在 PyTorch 里冻结参数。

### 13.1 DDP

PyTorch 入口按：

```text
local_batch_size = global_batch_size / world_size
```

分配数据。当前配置默认单卡 batch4；多卡时模型每卡完整复制，DDP 平均梯度，不是 FSDP。

### 13.2 Gradient checkpointing

默认开启，换取更低显存、更多重算。A100 实测：

```text
100 步 smoke: peak allocated 34.50 GB
```

显存不足时优先：

```text
--batch_size 2
--enable-gradient-checkpointing
多卡 DDP
```

不要把 `action_dim` 或 `action_horizon` 为省显存随意改小，否则训练协议和客户端不匹配。

---

## 14. 训练内评估

`train_pytorch.py` 每 `eval_interval` 步做：

```text
当前 train batch
→ sample_actions(num_steps=10)
→ 与同一 batch ground truth 计算 normalized MSE
```

还会额外跑一次 `num_steps=1`。

这是在线健康检查，不是验证集：

- 样本来自 train split；
- 可能刚参与训练；
- MSE 是 normalized action 空间；
- 不能作为最终 checkpoint 选择依据。

单任务实测：

```text
loss:         1.138 -> 0.024
train eval:   0.682 -> 0.009
```

最终质量应看验证 episode 的开环 MAE 和 SIMPLE 闭环成功率。

---

## 15. 验证集开环评估

[eval_openloop.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/eval_openloop.py)
直接读取验证 episode 的：

```text
states
action
egocentric video
```

对每隔 `stride` 帧调用一次策略，预测 `[30,36]`，再与录制动作做平均绝对误差。

误差按语义块统计：

```text
left_hand   [0:7]
right_hand  [7:14]
left_arm    [14:21]
right_arm   [21:28]
torso_rpy   [28:31]
height      [31]
base_cmd    [32:36]
```

这里的 `base_cmd` MAE 应接近 0，因为服务端已经把四个常量写死。若不是 0，说明输出 transform、
norm stats 或 checkpoint 版本有问题。

运行：

```bash
cd src/openpi4OpenHLM
PYTHONPATH=src python examples/simple_g1_dex3/eval_openloop.py \
  --checkpoint checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000 \
  --dataset-root "$HF_LEROBOT_HOME/local/g1_dex3_tabletop" \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --episodes 80 81 82 --stride 5
```

已记录的验证结果：

```text
checkpoint 1500 -> 2000 -> 3000
右手 MAE  0.0303 -> 0.0229 -> 0.0174
右臂 MAE  0.0453 -> 0.0424 -> 0.0327
```

---

## 16. Checkpoint 内容

每个 step 目录：

```text
checkpoints/<config>/<exp>/<step>/
├── model.safetensors
├── metadata.pt
└── assets/
    └── <asset_id>/
        └── norm_stats.json
```

`model.safetensors`

```text
36 维 action projection 扩展后的完整 PyTorch 模型
```

`metadata.pt`

```text
global_step
完整 TrainConfig
timestamp
has_ema
```

`norm_stats.json`

```text
训练时实际使用的 state/action q01/q99
```

服务端通过 `model.safetensors` 是否存在判断 PyTorch checkpoint，并从同一目录读取 norm stats 和
metadata，避免训练/服务统计版本错配。

当前 PyTorch 训练调用 `save_optimizer=False`，所以 AdamW moment 不保存在 checkpoint 中。
`--resume` 能恢复权重和 step，但 optimizer 状态会重置，不是 bit-exact 续训。

---

## 17. 服务端输出链

推理时：

```text
模型 raw action [30,36]
→ Unnormalize(norm_stats)
→ SimpleG1Dex3Outputs
   ├─ 截取前 36 维
   ├─ 检查非 NaN/Inf
   └─ 写回 height 和 base command 常量
→ WebSocket response["actions"]
```

这里 `SimpleG1Dex3Outputs` 在服务端执行，客户端不应该再假设模型输出是 normalized action。

---

## 18. SIMPLE 客户端代码

客户端是
[pi05_openhlm.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/simple_client/pi05_openhlm.py)。

它不是替换 SIMPLE 自带 baseline，而是新增一个独立类，保留原 baseline 作为对照。

### 18.1 输入组装

```text
observation["joint_qpos"]
→ 根据 STATE_SLICES 取手和臂
→ 拼接 last_cmd_torso_rpyh[4]
→ 32 维 state

observation["head_stereo_left"]
→ observation/image

instruction 或 DEFAULT_PROMPT
→ prompt
```

初始指令历史：

```text
[roll=0.0, pitch=-0.15, yaw=0.0, height=0.75]
```

每执行一条 `eval_move_actuators`：

```text
AMO command:
[vx, target_yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll, turning]

下一 state 的 4 个历史维:
[command[6], command[5], command[4], command[3] + 0.75]
= [roll, pitch, yaw, absolute height]
```

因此训练 state 和部署 state 使用同一段历史语义。

### 18.2 模型 action 到 SIMPLE 控制

先把模型顺序：

```text
[左hand7, 右hand7, 左arm7, 右arm7]
```

改为 `robot.joint_names[15:]` 要求：

```text
[左arm7, 右arm7, 左hand7, 右hand7]
```

左 hand 单独做：

```text
dataset [thumb, middle, index]
→ PD     [thumb, index, middle]
```

torso：

```text
模型 action[28:31] = roll,pitch,yaw
waist_qpos         = {"yaw": yaw, "roll": roll, "pitch": pitch}
AMO command        = [..., yaw, pitch, roll]
```

两种顺序不同是有意的。

### 18.3 Action chunk 排队与边界桥接

当内部队列为空时：

```text
发送一幅图 + 32 维 state + prompt
→ 服务端返回 [H,36]
→ 校验 shape/finite
→ 与上一块末条实际执行目标做 6 帧 smoothstep 边界桥接
→ 每行 split_action()
→ 转成 eval_move_actuators
→ 进入 PrimitiveAgent 队列
```

`chunk_blend_steps` 默认 6，也可用 `SIMPLE_CHUNK_BLEND_STEPS` 覆盖。桥接只改 action `0:31`
（手、臂、torso），保持新 chunk 的相对运动趋势；height 和离散/base command `31:36` 不混合。
它只处理边界位置突变，不是异步推理、latency compensation 或完整速度/加速度限制。

`upsample_factor` 可以把同一模型动作重复执行多次；`max_chunk` 可以只执行 chunk 前若干步。

### 18.4 延迟连接

SIMPLE 创建 agent 后会先做约 60 步 stand warmup。若构造函数一开始就建立 WebSocket，连接在第一次
推理前可能被 bundled websockets 客户端判为空闲并断开。

当前实现：

- `__init__` 只保存地址；
- 第一次 `infer()` 前调用 `_ensure_client()`；
- `reset()` 时主动关闭旧 socket。

### 18.5 Reset

每个新 episode：

```text
清空 action queue
清空 last chunk
重置 command history
强制重新连接
重置 global step
```

这防止上一 episode 的 height/rpy 或残余动作污染下一个 episode。

### 18.6 Handshake

服务端 metadata：

```json
{
  "action_protocol": "simple_g1_dex3",
  "action_dim": 36,
  "state_dim": 32,
  "action_horizon": 30
}
```

客户端检查 protocol 和 action_dim。若连到了原始 34 维 OpenHLM policy，会直接报错。
没有 metadata 的旧服务端只警告，不强制失败。

---

## 19. 测试体系

### 19.1 协议单测

```bash
PYTHONPATH=src python examples/simple_g1_dex3/test_protocol.py
```

无需 GPU 和数据集。检查：

- 32/36 维常量；
- 常量维是 `31:36`；
- 左手置换；
- `pin_constant_dims()` 不修改非恒定 torso；
- action 分块；
- 单视角 mask；
- 错误维度拒绝；
- 模型输出截断和非有限值拒绝。

测试还会在 `SIMPLE_ROOT` 可用时与 SIMPLE 的 `postprocess_psi0.py` / `pi05.py` 对照。

### 19.2 端到端数据链路

```bash
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" PYTHONPATH=src \
python examples/simple_g1_dex3/test_pipeline_e2e.py \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

验证：

```text
LeRobotDataset
→ 80 episode train split
→ action chunk [30,36]
→ 单视角 mask
→ 量化归一化
→ 224×224 resize
→ state/action pad 到 36
→ 归一化往返误差 < 1e-3
```

已记录结果：

```text
train split:       9,794 frames
round-trip error: 2.22e-16
```

### 19.3 多任务链路

```bash
PYTHONPATH=src python examples/simple_g1_dex3/test_multitask_pipeline.py
```

检查：

- `ConcatDataset` 的帧数等于 manifest `frames_train`；
- 子数据集数量和 manifest 一致；
- 每个子数据集 state/action shape 正确；
- 三个 prompt 彼此不同；
- 归一化后 image 224×224、state 36、action `[30,36]`。

---

## 20. 从数据到训练的完整命令

以下命令均在 `src/openpi4OpenHLM` 执行。

### 20.1 数据

```bash
export HF_LEROBOT_HOME=/data/psi_simple_tabletop/lerobot

python examples/simple_g1_dex3/audit_data.py \
  --dataset-root "$HF_LEROBOT_HOME/local/g1_dex3_tabletop" \
  --check-videos

python examples/simple_g1_dex3/make_split.py \
  --dataset-root "$HF_LEROBOT_HOME/local/g1_dex3_tabletop" \
  --out assets/simple_g1_dex3/tabletop_split.json \
  --val-fraction 0.2 --seed 0
```

### 20.2 环境与 base 权重

一键脚本：

```bash
bash examples/simple_g1_dex3/deploy_server.sh env
bash examples/simple_g1_dex3/deploy_server.sh weights
```

`weights` 会下载并校验 PyTorch `pi05_base/model.safetensors`。

### 20.3 统计

```bash
PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_tabletop \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

### 20.4 冒烟训练

```bash
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py pi05_simple_g1_dex3_tabletop \
  --exp_name smoke \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --batch_size 1 --num_train_steps 100 \
  --save_interval 50 --log_interval 10 \
  --no-wandb-enabled --overwrite
```

重点观察：

- loss 和 grad norm 是否有限；
- backward/optimizer 是否有一次正常执行；
- 显存是否在预期范围内；
- `50/`、`100/` checkpoint 是否完整；
- checkpoint 输出是否为 `[30,36]`。

### 20.5 正式训练

```bash
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py pi05_simple_g1_dex3_tabletop \
  --exp_name run1 \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --no-wandb-enabled
```

恢复：

```bash
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py pi05_simple_g1_dex3_tabletop \
  --exp_name run1 \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
  --no-wandb-enabled --resume
```

---

## 21. 多任务训练命令

先生成 manifest：

```bash
python examples/simple_g1_dex3/make_multitask_manifest.py \
  --repo local/g1_dex3_tabletop \
  --repo local/g1_dex3_bend_pick_mp \
  --repo local/g1_dex3_bend_pick_teleop \
  --out assets/simple_g1_dex3/multitask_manifest.json \
  --seed 0 --val-fraction 0.2
```

计算多任务统计：

```bash
PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_multitask
```

训练：

```bash
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train_pytorch.py pi05_simple_g1_dex3_multitask \
  --exp_name multitask_run1 \
  --no-wandb-enabled
```

必须先确认 manifest 中的 repo_id 和实际 LeRobot 目录完全一致。

---

## 22. 当前实现边界

### 22.1 闭环成功不等于协议全正确

当前 3000 步 checkpoint 在官方 eval 包上的结果是：

```text
DR0: 70%
DR1: 70%
DR2: 40%
总计: 18/30 = 60%
```

成功说明基本动作链路可以工作，但不应掩盖以下仍需持续检查的内容：

- 手部顺序；
- torso rpy 到 waist/AMO 的双顺序；
- 历史 state 是否严格对应上一条指令；
- checkpoint 与 client metadata 是否一致。

### 22.2 训练内评估不是验证

训练内 MSE 使用当前 train batch，只能检查在线生成是否稳定。checkpoint 选择应使用验证 episode 的：

- 分块开环 MAE；
- SIMPLE 闭环成功率；
- 官方 eval 包 DR 分级结果。

### 22.3 `base_cmd` 被常量写死

当前数据集里 vx/vy/turning/target_yaw 全为 0，所以服务端直接写回 0。若以后训练移动任务，这五个
常量以及 `pin_constant_dims()` 必须一起修改；否则模型即使学会移动，也会在输出阶段被覆盖。

### 22.4 多任务权重按帧数自然加权

`ConcatDataset` 只是拼接，没有任务均衡采样。三任务帧数差异会直接影响采样比例。

### 22.5 多任务 manifest 哈希尚未在加载时校验

生成端有 `sha256`，加载端当前未校验。严格复现时应补上。

### 22.6 PyTorch 不是 bit-exact resume

checkpoint 未保存 optimizer state、sampler 和 RNG 状态。恢复后会重置 AdamW moment。

### 22.7 PyTorch 冻结配置不生效

`freeze_filter` 当前主要用于 JAX 路径；PyTorch 主循环没有根据它设置 `requires_grad`。

### 22.8 多任务 prompt 需要客户端正确传入

训练时每个 repo 的 LeRobot task 会成为 prompt。部署多任务策略时，客户端必须给当前任务对应指令，
不能依赖 `policy_metadata` 中单一的 tabletop prompt。

---

## 23. 建议的代码阅读顺序

按数据流阅读最容易理解你的实现：

1. [simple_g1_dex3_policy.py](../src/openpi4OpenHLM/src/openpi/policies/simple_g1_dex3_policy.py)
   - 32/36 维协议；
   - 常量维；
   - `SimpleG1Dex3Inputs/Outputs`；
   - `pin_constant_dims()`。
2. [config.py](../src/openpi4OpenHLM/src/openpi/training/config.py)
   - `LeRobotSimpleG1Dex3DataConfig`；
   - `pi05_simple_g1_dex3_tabletop`；
   - `pi05_simple_g1_dex3_multitask`。
3. [audit_data.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/audit_data.py)
   - 数据到底包含什么；
   - 哪些检查会在训练前失败。
4. [make_split.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/make_split.py)
   - 场景组级 train/val 隔离。
5. [manifests.py](../src/openpi4OpenHLM/src/openpi/training/manifests.py)
   - manifest 校验和注入 config。
6. [data_loader.py](../src/openpi4OpenHLM/src/openpi/training/data_loader.py)
   - `EpisodeSubsetDataset`；
   - `ConcatDataset`；
   - `delta_timestamps`。
7. [test_pipeline_e2e.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/test_pipeline_e2e.py)
   - 用真实数据观察每个 transform 后的 shape。
8. [train_pytorch.py](../src/openpi4OpenHLM/scripts/train_pytorch.py)
   - `load_model_with_action_proj_expansion`；
   - `train_loop`。
9. [pi0_pytorch.py](../src/openpi4OpenHLM/src/openpi/models_pytorch/pi0_pytorch.py)
   - `action_proj_dim=36`；
   - `embed_suffix`；
   - `forward`；
   - `sample_actions`。
10. [eval_openloop.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/eval_openloop.py)
    - 验证集如何按动作块解释误差。
11. [pi05_openhlm.py](../src/openpi4OpenHLM/examples/simple_g1_dex3/simple_client/pi05_openhlm.py)
    - state 历史；
    - hand permutation；
    - `[roll,pitch,yaw]` 到 waist/AMO 的两套顺序；
    - reset 与 WebSocket 生命周期。

---

## 24. 一页速查

```text
数据:
  states 32
  action 36
  50 fps
  action_horizon 30 = 0.6 s
  单视角 640×360 -> 224×224

模型:
  π0.5 base PyTorch
  action_dim 36
  state 32 离散进 prompt
  action expert 连续 flow matching

权重:
  前 32 维动作投影复制 base
  后 4 维 Xavier 初始化
  输出 bias 后 4 维置零

优化:
  AdamW
  peak lr 3e-5
  warmup 300
  cosine decay 到 3e-6
  3000 steps
  batch 4
  bf16
  全参数微调

常量:
  action[31] = 0.75
  action[32:36] = 0

客户端:
  state = proprio11 + previous command rpyh
  left hand permutation once
  torso rpy PD: roll,pitch,yaw
  torso rpy AMO: yaw,pitch,roll
```

这套代码的关键不是单独某个模型函数，而是“训练协议、数据 split、归一化、36 维动作头、服务端
常量写回和 SIMPLE 客户端”六者必须保持一致。任何一处顺序或尺度改变，都应从 `test_protocol.py`
和真实 checkpoint 推理开始重新验证。

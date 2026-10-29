# G1 + 双 Dex3-1：π0.5 全量微调与 SIMPLE 单任务闭环方案

> 2026-10-29 仓库精简说明：本文保留早期方案和已有实验记录，其中旧组件路径为历史上下文。
> 当前源码只保留 G1 Dex3 数据/训练/评估/SIMPLE 部署与 Rerun；现行命令见
> [使用说明](../src/openpi4OpenHLM/examples/simple_g1_dex3/README.md)，回放模型移至 `assets/g1/`。

- 制定日期：2026-10-27（Asia/Shanghai）
- 状态：技术方案 + 第一阶段代码已在 A100 服务器完成环境部署、数据链路验证和训练冒烟（见第 13 节）；未在 SIMPLE/Isaac 中闭环，未连接真实机器人。
- 第一阶段目标：单任务、固定数据协议、全参数微调、SIMPLE 闭环验证。
- 不包含：HEFT 接入、SONIC 接入、LoRA、多任务训练、真实机器人运行。
- 后续扩展：在仿真基线成立后，再进行真实 G1 数据采集、微调和受控部署。

## 1. 锁定技术选型与验收范围

| 项目 | 第一阶段选择 |
|---|---|
| 机器人 | G1 身体 29 关节 + 左右 Dex3-1 各 7 关节 |
| 仿真任务 | `simple/G1WholebodyTabletopGraspMP-v0` |
| 实际任务目标 | 右手抓取桌面目标并抬升；不是双手搬袋，也不是完整桌面整理 |
| VLA | π0.5 base 初始化，使用 OpenHLM 内 openpi 分支进行全参数微调 |
| 动作协议 | PSI/SIMPLE 的 36 维指令，不使用 OpenHLM 原始 34 维协议 |
| 状态协议 | 32 维：双手 14 + 双臂 14 + 上一执行指令的躯干 RPY/高度 4 |
| 视觉输入 | 单个第一视角；不要求新增腕部相机 |
| 底层执行 | SIMPLE 原有 AMO + 双臂/双手关节控制，复用现有环境执行逻辑 |
| 模型服务 | OpenHLM WebSocket policy server |
| 仿真客户端 | 基于 SIMPLE 现有 Pi05Agent 新增独立适配版本，避免覆盖原 baseline |

这里复用的是 OpenHLM 的 π0.5 训练与服务能力，不是复现其“SONIC + 双夹爪 + 34 维全身参考动作”的原始配置。全参数微调与采用 AMO 并不冲突。

### 验收等级

1. **工程闭环**：正确数据可加载、全参数更新可验证、checkpoint 可保存/加载、模型动作可进入仿真、episode 正常结束并产生日志。
2. **固定场景能力**：在少量预先定义的场景中，模型能重复完成抓取抬升；单次成功只说明基本链路成立。
3. **泛化评测**：按冻结的验证/评测协议报告成功率、失败原因和时序指标；不预先承诺成功率。
4. **真实部署能力**：不属于第一阶段交付，不能用仿真成功替代实机验证。

## 2. 总体架构

```text
匹配的 SIMPLE 示范数据（训练子集）
    → 数据/图像/状态/动作适配
    → 只用训练子集计算归一化统计
    → π0.5 base 的全参数微调
    → checkpoint + 同版本统计 + 协议元信息
    → 远程 WebSocket policy server

SIMPLE 环境
    → 第一视角 RGB + 按约定组装的 32 维状态 + 任务指令
    → policy server
    → (H, 36) 未归一化动作序列
    → SIMPLE 专用客户端
        ├─ 双 Dex3 / 双臂关节目标
        └─ AMO 指令及现有腰部目标处理
    → 环境执行、观测更新、成功/失败评估
```

第一阶段不启动 OpenHLM 原来的真实机器人部署启动器，不启动 HEFT 或 SONIC。所有控制仅作用于仿真。

## 3. 数据与实验划分

### 3.1 使用正确数据

- 训练数据：[G1WholebodyTabletopGraspMP-v0.zip](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/blob/main/simple/G1WholebodyTabletopGraspMP-v0.zip)
- 评测包：[simple-eval/G1WholebodyTabletopGraspMP-v0.zip](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/blob/main/simple-eval/G1WholebodyTabletopGraspMP-v0.zip)

已检查训练包元信息：LeRobot v2.1；100 episodes；12,226 帧；50 FPS；第一视角 640×360；state 32 维、action 36 维。图像视频使用 AV1，需在最终数据加载环境中验证解码与帧索引。

原始数据保持只读。建议使用 `/home/pc/桌面/OpenHLM/data/psi_simple_tabletop/` 作为未来数据根目录，区分下载包、原始训练数据、原始评测包、split manifest 和统计。该路径在本次仅作规划，不代表数据已下载。

OpenHLM 当前 LeRobot loader 按 `repo_id` 和 `LEROBOT_HOME` 定位数据。实施时可设 `LEROBOT_HOME=/home/pc/桌面/OpenHLM/data/psi_simple_tabletop/lerobot`，为完整数据副本指定 repo_id `local/g1_dex3_tabletop`，使数据实际位于 `/home/pc/桌面/OpenHLM/data/psi_simple_tabletop/lerobot/local/g1_dex3_tabletop/`。训练 split 用 episode manifest 过滤；评测仍保留原始环境元信息。若采用新增显式 root 配置，须单独实现和测试，不能把绝对目录当成现有 repo_id 接口已支持的能力。

### 3.2 划分规则

- 初始建议 80 个 episode 训练、20 个 episode 验证，固定 seed 并保存清单。
- 先检查场景/物体/初始配置是否重复；有重复时按场景组划分，不让高度相似配置跨集合泄漏。
- 禁止随机拆相邻帧。
- 仅训练集合参与梯度更新和归一化统计计算；验证集合用于调参和选 checkpoint。
- 官方评测包用于最终固定协议评测。先核对包内结构、split 含义以及与训练场景的重叠程度，不仅凭文件夹叫 eval 就声称完全未见场景。
- 保留 `environment_config`、episode 索引及资源 ID。SIMPLE 重建场景需要这些数据，不能只复制 parquet 和视频后丢掉元信息。
- 通过新增 episode 选择能力让训练和统计共用同一个 manifest；不要手动删文件破坏 LeRobot 索引。
- 如果现有评测 CLI 不接受 episode 列表，增加 manifest 支持，或逐个指定原始 episode index 后汇总结果。

### 3.3 数据检查清单

- 字段、维度、dtype、数值范围、NaN/Inf、每个 episode 的长度。
- 视频可解码、RGB/BGR、图像方向、对应帧是否准确。
- 动作时序、上一执行命令状态、初始姿态是否一致。
- 模型 action chunk 不跨 episode；结束补齐策略明确并纳入测试。
- 统计检查常量维度。第一轮沿用并验证底层 LeRobot 的边界处理；若改为停止型补齐，应保持绝对关节/高度/姿态/航向，速度及转向标志置零，并作为有记录的策略变更。

## 4. 冻结状态和动作协议

### 4.1 模型观测

| 字段 | 约定 |
|---|---|
| `observation/image` | 单个第一视角 RGB，运行时与训练使用一致 resize-with-pad 到 224×224 |
| `states` | float32，形状 `(32,)` |
| `prompt` | `grasp the object on the tabletop.`，训练和评测一致 |
| `reset` | episode 重置标识；客户端同时清空动作队列和历史命令 |

模型内部：`base_0_rgb` 有效；左右腕部图像占位且 mask=False。不要把缺失腕部图像置零后仍标记为有效，也不要用头图复制为腕图。

状态的最后四维是历史指令里的 RPY/高度，不应直接替换成当前 IMU RPY 或测得根部高度。实测状态与指令状态不是同一个量。模型内部可按 action_dim padding，但外部协议依然为 32 维。

### 4.2 模型动作

| 索引 | 语义 |
|---|---|
| 0:7 | 左手 Dex3，按数据的专用顺序 |
| 7:14 | 右手 Dex3，按数据顺序 |
| 14:21 | 左臂 7 关节目标 |
| 21:28 | 右臂 7 关节目标 |
| 28:31 | 躯干/腰部 roll、pitch、yaw 目标，沿用 SIMPLE 现有转换 |
| 31 | 绝对高度目标，米 |
| 32:34 | vx、vy 指令 |
| 34 | 转向标志，不当作 yaw 角速度 |
| 35 | 目标航向，按 SIMPLE 的既有参考约定 |

保持字段原语义，不进行统一的 joint delta 转换。输出为未归一化的 `(H,36)`，不能截断成 34 维。

### 4.3 手指顺序：必须专门测试

SIMPLE 的实际后处理代码对左手使用 thumb → middle → index；机器人 canonical 顺序则是 thumb → index → middle。PSI 某些文字注释与此并不一致，不能凭注释写映射。

已检查的参考实现：

- `/home/pc/桌面/loco-manipulation/opensource/SIMPLE/scripts/postprocess_psi0.py`
- `/home/pc/桌面/loco-manipulation/opensource/SIMPLE/src/simple/baselines/pi05.py`

左手数据的 0:3 是 thumb、3:5 是 middle、5:7 是 index。采用按关节名构建的映射，并用真实数据重建以及单关节脉冲测试确认：正向映射再逆映射不改变数值，左右手不串线。

### 4.4 已识别的客户端缺口

当前 SIMPLE Pi05Agent 先构造 32 维状态，但向服务端发送 `states[0, :28]`。第一阶段拟使用完整 32 维训练，因此必须修正这一不一致。

建议新增独立的 `pi05_openhlm` 客户端，而不是覆盖现有 `pi05` baseline。新客户端复用已有关节/AMO 指令转换，发送完整状态，并补上形状、有限值、reset、动作队列和协议检查。

OpenHLM 原 `G1Outputs` 固定截取 34 维，原机器人客户端也按 34 维解包；本项目均不复用这两项。

## 5. 环境与算力安排

### 5.1 本机实际情况

本次只读探测到：RTX 4060 Laptop，约 8GB 显存；约 32GB 系统内存；工作分区剩余约 848GB。

- 不将本机作为默认全参数训练设备，也不承诺本机能同时运行 π0.5 与 Isaac Sim。
- 本机先做源码适配、数据检查和单环境仿真冒烟。
- 8GB 显存运行目标 SIMPLE/Isaac 场景是否足够需要实测；失败时换到有更充足显存的 RTX 机器，不通过偷偷切换任务/渲染模式来规避验收。

### 5.2 推荐分工

| 资源 | 用途 |
|---|---|
| 单卡 A100/H100 80GB 级远程 GPU | 小 batch 全参数训练；训练后复用该卡运行推理服务 |
| 本机 RTX 4060 Laptop | 尝试单环境仿真、客户端、开发；不保证目标场景可承载 |
| 可选远程 RTX 24GB 或更高显存机器 | 本机仿真资源不足时替代；属于容量建议，不是实测最低门槛 |

A100/H100 训练能力不等于适合 SIMPLE 的 RTX 渲染要求。训练、服务与仿真可分机器，不必同一台运行。

### 5.3 两套独立环境

- OpenHLM 当前训练分支：Python >=3.11、torch 2.9.0、transformers 4.53.2，以及仓库自己的 transformers patch。
- 当前 SIMPLE：Python 3.10、torch 2.7.0、Isaac Sim 4.5.0、MuJoCo 3.3.6。
- 分别建立虚拟环境，不交叉安装两个 openpi fork，不把 SIMPLE 依赖强塞进训练环境。
- 两端只通过兼容的 openpi-client/WebSocket 协议通信。
- 固定 Git commit、基础模型版本、LeRobot 版本、环境锁文件、数据包校验和及场景资源版本。
- WebSocket 默认走 localhost/SSH 隧道或受控内网，不直接暴露到公网。

## 6. 分阶段实施与门槛

| 阶段 | 工作 | 必须产出/通过条件 |
|---|---|---|
| M0 基线冻结 | 固定代码版本、任务、数据、模型基座、依赖与资产清单 | 可审计版本清单；AMO 权重和场景资源可获得；原始数据可读 |
| M1 数据与协议 | 检查数据、建立 split、实现单视角/32-state/36-action 适配 | 数据及映射单测通过；无跨 episode chunk；训练/推理归一化往返通过 |
| M2 无 VLA 回放 | 原始示范动作 + SIMPLE/AMO，在保存的环境配置下回放 | 至少建立可重复的成功示范回放；场景、左右手及初始状态一致 |
| M3 训练冒烟 | 100–300 步全参数训练并保存/重载 | loss/梯度有限、代表性模块梯度可检查、显存可控、checkpoint 与统计完整 |
| M4 小样本拟合 | 从训练集选 2–5 episodes 做短程拟合 | 开环误差降低且动作可视化合理；同场景闭环开始出现成功；不是泛化指标 |
| M5 单任务训练 | 使用训练子集训练，验证集选 checkpoint | 保存每次配置、验证误差、动作范围、训练吞吐与模型版本 |
| M6 仿真闭环 | 固定场景 → 验证场景 → 官方评测协议 | 成功率分子/分母、失败分类、视频与时序统计；评测数据不回流选模型 |
| M7 可选实机扩展 | 标定、少量真实示范、再微调、受控部署 | 独立安全验收；不能从仿真直接跳到无人值守运行 |

关键原则：M2 的示范回放失败时，优先排查资产版本、场景重置、控制映射和时间轴，不用增加训练步数掩盖接口问题。

## 7. 训练方案

### 7.1 基座和参数更新

- 第一版固定为 π0.5 base；若后续改为 π0.5-DROID，视作独立实验，不与 base 结果混用。
- 不从 OpenHLM 原 34 维任务 checkpoint 随意续训来替代基座适配。
- 使用 OpenHLM 已有的 action projection 扩展逻辑，从预训练维度适配到 36 维；检查加载结果与未加载参数清单。
- 不使用 LoRA，不冻结主干。记录总参数量、requires_grad 参数量，以及视觉/语言/动作专家代表层的梯度；未参与前向的参数单独解释，不能要求每个参数每步都非零梯度。
- 保留绝对关节目标和各控制量原本语义；重新计算本任务训练子集的归一化统计。

### 7.2 建议起始配置（待实现、待实测，不是作者复现实测配方）

| 配置 | 初始建议 |
|---|---|
| 配置名 | `pi05_simple_g1_dex3_tabletop` |
| `pi05` | true |
| `action_dim` | 36 |
| `action_horizon` | 30，与 Psi0 该任务配置对齐；50Hz 下为 0.6 秒参考动作 |
| `discrete_state_input` | true |
| 图像 | 单视角 224×224；两腕 mask=False |
| 精度 | BF16 |
| batch size | 单卡从 2 或 4 测峰值显存，再尝试 8；不直接照搬 128/256 |
| gradient checkpointing | 开启 |
| training compile | 冒烟先关闭；稳定后单独评估 |
| EMA | 第一版关闭 |
| optimizer | 先用普通 AdamW，便于排错 |
| workers | 先用 2–4，按视频解码/内存压力调整 |
| 峰值学习率 | 可从 3e-5 起步，作为待验证超参数；不声称是最优值 |
| 首轮正式预算 | 约 3,000 步起步，结合验证决定是否延长至 5,000–10,000，而非必跑 30,000 |
| warmup / 保存 | 3,000 步实验可先 warmup 100、每 500 步保存；随总预算一致调整 schedule |
| 服务端采样 | 先沿用 10 个去噪步，再测速度与效果 |

少数据不等于低显存；小任务主要减少数据准备和所需实验量。当前 PyTorch 主入口采用 DDP，不是 FSDP/ZeRO；未发现现成梯度累积实现，不能把它写成已经可用的命令选项。

先排除编译/首轮加载时间，实测平均 step 时间和显存，再估算完整 GPU·小时。只读 metadata 的时长不是训练耗时。

## 8. 仿真验证与时序

### 8.1 回放先行

- 固定一个原始 episode 的 scene/object/robot/camera 配置。
- 使用记录动作通过与模型相同的解包及 AMO 执行链回放，不用直接设置 qpos 的运动学播放替代物理验证。
- 确认 AMO 权重、关节模型、Dex3 接触、物体资产和 PD 配置一致。
- 核对 stand warmup；模型与回放均采用同一 reset/预站立约定，避免重复 warmup 或把它混入评价时间。

### 8.2 开环指标

- 头部图像、状态与预测动作可视化。
- 分组误差：左手、右手、左臂、右臂、RPY、高度、移动指令、转向。
- 不只报告一个把弧度、米、速度、离散标志混合的总 MSE。
- 检查归一化后的有效范围、反归一化、输出有限性和关节限位。

### 8.3 闭环指标

1. 训练场景闭环只记为拟合/链路检查。
2. 验证场景用于开发；模型定版后再做固定最终评测。
3. 记录每个 episode 的成功、失败、超时、跌倒、抓空、滑落、模型越界和接口异常。
4. 区分环境/解码/网络错误与正常任务失败，报告有效回合及异常回合数量，不能静默排除失败来抬高成功率。
5. 保存视频、动作、reset 配置、模型和协议版本。
6. 当前任务 reward 由物体抬升量计算，`_LIFT_HEIGHT=0.02m`，原 `check_success` 的阈值为 reward>=0.9。报告应注明约 1.8cm 抬升对应这一判定，不能将它描述为“完成放置”。额外加入持稳时间等标准时，必须单列指标，不悄悄替换原 benchmark 成功定义。

### 8.4 频率与 action chunk

- 数据与当前 AMO 基线按 50Hz 约定对齐，不沿用 OpenHLM 实机例子的 30Hz。
- 初始 H=30、upsample factor=1，先复用客户端执行完整 chunk 的行为，便于与源 baseline 对照。
- 后续可以只执行前 5–10 步就重新规划，但需要新增明确配置并作为独立实验，不当成现有默认功能。
- 同步仿真在请求推理时可以等待，因此墙钟速度慢不等于仿真步长变化。分别记录仿真时间、墙钟时间、RTF、推理 P50/P95 延迟和客户端等待时间。
- 仿真成功不能证明实机满足实时性；真实部署需要底层控制独立持续运行、动作缓冲、超时回退与停止逻辑。

## 9. 代码改动（已实现骨架，见第 13 节）

优先保持改动隔离，不更改原 OpenHLM G1/SONIC 路线和 SIMPLE 原 pi05 baseline。

| 文件/位置 | 拟改动 |
|---|---|
| `/home/pc/桌面/OpenHLM/src/openpi4OpenHLM/src/openpi/policies/simple_g1_dex3_policy.py`（新增） | 单视角、32 维状态、36 维输出转换 |
| `/home/pc/桌面/OpenHLM/src/openpi4OpenHLM/src/openpi/training/config.py` | 新数据配置和训练配置、明确 action_sequence_keys、episode 选择配置 |
| `/home/pc/桌面/OpenHLM/src/openpi4OpenHLM/src/openpi/training/data_loader.py` | 可选 episode manifest/indices 过滤；默认行为不变，统计与训练共用 |
| `/home/pc/桌面/OpenHLM/src/openpi4OpenHLM/examples/simple_g1_dex3/`（新增） | 数据审计、split、模型冒烟、开环评估、协议测试工具及说明 |
| `/home/pc/桌面/loco-manipulation/opensource/SIMPLE/src/simple/baselines/pi05_openhlm.py`（新增） | 独立客户端，完整 32-state、正确手指映射、输出检查及 reset |
| `/home/pc/桌面/loco-manipulation/opensource/SIMPLE/src/simple/cli/eval.py`（按需） | 仅在现有接口不足时增加 manifest 选择能力；先评估可否用外部 runner |

配置重命名、统计资产 ID、checkpoint 路径及服务元信息必须一致。checkpoint 包含模型、配置、norm stats、split manifest/hash、动作协议版本；禁止训练后临时换一份数据统计再部署。

必要自动化测试：

- 32/36 维度、单视角 mask、图像输入范围和通道顺序。
- 字段 repack 与 `action` 时间序列取样；episode 边界。
- 模拟观测→模型状态与数据状态重建的一致性。
- Dex3 正逆映射、左右手、腰部 RPY 顺序。
- 反归一化与常量维度；NaN/Inf/错误形状拒绝。
- checkpoint 保存/重载与 36 维投影形状。
- reset 清空动作队列、命令历史和 episode 状态。
- WebSocket request/response 冒烟；不依赖真实硬件。

## 10. 实机扩展路线（仿真通过后另行执行）

```text
确认 G1/双 Dex3 与 AMO 实机接口
    → 对齐第一视角相机、关节顺序、标定和控制限位
    → 真实状态/图像只读与动作记录验证
    → 受控采集同一抓取任务的少量真实示范
    → 转为同一 32-state / 36-action 协议，重新核对历史命令语义
    → 真实数据再微调或受控混合训练
    → 在独立安全验收下进行逐级实机测试
```

- 不将原 OpenHLM 的双夹爪采集脚本直接当作 Dex3 采集系统。
- 仿真策略不能未经验证直接零样本部署到机器人。
- 与机器人端的模型、控制周期、手指零位、质量/负载及关节限位均需核对。
- 有监督操作、急停、保护空间、超时处理和安全复位方案；本阶段不会发送任何真实机器人控制命令。
- 实机数据格式相同并不意味着视觉域、摩擦、相机几何和动力学已一致；需单独检查 sim-to-real 差异。

## 11. 风险优先级与决策规则

| 风险 | 处理规则 |
|---|---|
| 本机 8GB 无法承载目标 Isaac 场景 | 优先迁移仿真至 RTX 大显存机器；训练保持远程 |
| 训练/推理 state 32 vs 28 不一致 | 必须先修复并单测，再训练 |
| 左手 index/middle 顺序错误 | 数据重建 + 正逆映射 + 单关节验证，禁止靠抓取成功偶然验证 |
| 缺少环境配置或资产版本不匹配 | 先恢复场景并回放示范，暂停 VLA 调参 |
| 原始成功判定过于宽松 | 原指标照报，附加稳定抓持等指标单列 |
| 小数据拟合强、验证差 | 检查泄漏、分布和场景覆盖，再调学习率/步数或补数据 |
| 推理延迟高 | 先区分模型、网络、解码和同步等待；不直接声称是 AMO 不稳 |
| 有训练 loss 但闭环全失败 | 优先核对协议、重置、图像和回放，不能只延长训练 |

## 12. 最先执行的工作包

1. 下载并审计匹配的训练包和评测包，冻结 split 与资产清单。
2. 建立 SIMPLE 单环境及示范回放基线，测本机显存是否足够。
3. 实现 PSI/SIMPLE 专用模型适配和独立客户端，优先解决 32-state 与手指顺序问题。
4. 通过协议测试后，再安排远程 80GB GPU 做短程全参数训练。

## 13. 已实现的代码与验证边界

| 文件 | 状态 |
|---|---|
| `src/openpi4OpenHLM/src/openpi/policies/simple_g1_dex3_policy.py` | 已实现：单视角、32-state、36-action、常量维写回、手部置换常量 |
| `src/openpi4OpenHLM/src/openpi/training/config.py` | 已实现：`LeRobotSimpleG1Dex3DataConfig`、`pi05_simple_g1_dex3_tabletop`、`episode_manifest` |
| `src/openpi4OpenHLM/src/openpi/training/data_loader.py` | 已实现：`EpisodeSubsetDataset` + `episode_frame_indices`，默认行为不变 |
| `src/openpi4OpenHLM/src/openpi/training/manifests.py` | 新增：split manifest 读写与 sha256 校验 |
| `src/openpi4OpenHLM/examples/simple_g1_dex3/` | 新增：`audit_data.py`、`make_split.py`、`test_protocol.py`、`eval_openloop.py`、`README.md` |
| `SIMPLE/src/simple/baselines/pi05_openhlm.py` | 新增：`Pi05OpenhlmAgent`，完整 32-state、36-dim 输出、元信息校验、reset |

已实际验证：

- 全部文件 `py_compile` 通过；`audit_data.py` 跑通发布包全部 100 episode（12,226 帧）。
- 发布包实测：`action[31]` 恒 0.75、`action[32:36]` 全 0；**`action[28:31]`（torso rpy）不是常量**，会漂移约 0.05 rad，因此第 9 节的常量维是 31–35 而不是 28–35。
- 手部顺序用真实数据交叉验证：左手 dataset 顺序为 thumb, middle, index，右手为 thumb, index, middle；客户端只对左手做 `(0,1,2,5,6,3,4)` 置换，**不再调用 `from_psi0_upper_joints`**（该函数会二次置换左手）。
- 动作块顺序实测为**手在前（0:14）、臂在后（14:28）**，与 OpenHLM 34 维协议的“臂前手后”相反；文档第 4.2 节已按实测修正。

### 13.1 A100 服务器实测结果

部署目标：`openhlm-server`，A100-SXM4-80GB，驱动 570 / CUDA 12.8，Python 3.11。
项目目录 `~/OpenHLM_src/openpi4OpenHLM`，独立 `.venv`，不污染系统环境。

已通过：

| 检查 | 结果 |
|---|---|
| 环境导入 | torch 2.9.0+cu128、jax 0.5.3 CUDA、transformers 4.53.2 |
| 协议单测 | `test_protocol.py` 9/9 |
| 数据审计 | 100 episodes / 12,226 frames / 50 fps；state 32、action 36 |
| 数据链路 | `test_pipeline_e2e.py` 通过；train split 80 episodes / 9,794 frames；归一化往返最大误差 `2.22e-16` |
| split | `assets/simple_g1_dex3/tabletop_split.json`，sha256 `42768e07...` |
| 归一化统计 | `assets/pi05_simple_g1_dex3_tabletop/local/g1_dex3_tabletop/norm_stats.json`，state 32 / action 36 |
| WebSocket 冒烟 | metadata `action_protocol=simple_g1_dex3, action_dim=36`；返回 `(30,36)` |
| 100 步全参训练 | loss 0.76 → 0.58；峰值 allocated 34.50 GB / reserved 35.21 GB；稳定约 1.5 step/s |
| checkpoint | `checkpoints/pi05_simple_g1_dex3_tabletop/smoke/50` 和 `/100`，每个含 model、metadata、norm stats |
| checkpoint 推理 | 直接加载 100 步 checkpoint 返回 `(30,36)`；首轮含编译 19.7 s，热推理 398 ms/chunk |
| 3,000 步全参训练 | loss 1.138 → 0.024；训练内 eval MSE 0.682 → 0.009 |
| 最终 checkpoint | `checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000`；500/1000/1500/2000/2500/3000 全部保留 |
| 验证集离线 MAE（5 episodes） | 1,500 → 2,000 → 3,000：右手 `0.0303 → 0.0229 → 0.0174`，右臂 `0.0453 → 0.0424 → 0.0327` |
| 最终 WebSocket 服务 | 3000 checkpoint 已加载，metadata 正确；首轮 17.5 s，热请求 392 ms/chunk |

部署过程中修正的、会影响真实训练的问题：

- LeRobot 新版把 `LEROBOT_HOME` 设为硬报错并改名为 `HF_LEROBOT_HOME`，脚本和文档已统一。
- 数据列名是 `action` 而非默认的 `actions`，已在 `action_sequence_keys` 显式固定。
- `pyarrow>=21` 与 `numpy<2` 冲突，锁定 `pyarrow>=15,<21`；`torchcodec>=0.10` 依赖 CUDA 13 的 `libnvrtc.so.13`，在 CUDA 12.8 环境锁定 `<0.10`。
- OpenHLM 的 transformers patch 必须显式复制，已加入 `deploy_server.sh env`。
- 训练脚本原先不展开 `pytorch_weight_path` 的 `~`，已修复。
- `PI0Pytorch` 构造函数无条件对采样路径做 `max-autotune`，训练会被无谓阻塞；现在只在服务端 `create_trained_policy` 编译采样路径。
- 服务端输出在反归一化后强制写回 height 和四个底座常量，客户端再做一次幂等保护。

**仍未验证**：SIMPLE/Isaac 中的真实闭环、AMO 回放和官方评测集；用户已将这部分安排到另一张 GPU 上执行。`pi05_openhlm` 客户端的完整 SIMPLE run 仍需在那边联调。A100 上的最终策略服务保持运行在 `0.0.0.0:8000`，加载 `run1/3000`。

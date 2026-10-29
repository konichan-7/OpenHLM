# G1 Dex3 归一化统计

训练与服务使用相同的状态/动作统计。G1 Dex3 的状态为 32 维，动作为 36 维，
使用量化归一化；统计只来自训练 split。运行目录为 `src/openpi4OpenHLM`。

## 单任务

先按 [数据划分说明](../examples/simple_g1_dex3/README.md#4-冻结-split) 生成清单，
再计算统计：

```bash
export OPENPI_DATA_HOME="$PWD"
PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_tabletop \
  --episode-manifest assets/simple_g1_dex3/tabletop_split.json
```

默认写入 `assets/pi05_simple_g1_dex3_tabletop/local/g1_dex3_tabletop/norm_stats.json`，包含 `state` 和
`actions` 两组统计。训练使用同一份 `--episode-manifest`。

## 多任务

按 [训练代码导读](../../../docs/g1_dex3_training_code_walkthrough.md#21-多任务训练命令)
生成 `assets/simple_g1_dex3/multitask_manifest.json` 后执行：

```bash
PYTHONPATH=src python scripts/compute_norm_stats.py \
  --config-name pi05_simple_g1_dex3_multitask
```

默认写入 `assets/pi05_simple_g1_dex3_multitask/g1_dex3_multitask/norm_stats.json`，
统计覆盖 manifest 中选定的训练 episodes。可通过配置中的 assets 路径覆盖位置。

checkpoint 保存关联统计；服务使用与该 checkpoint 匹配的统计。数据、统计和
checkpoint 是本地实验产物，不提交到 Git。任务常量维由
`SimpleG1Dex3Outputs` 在反归一化之后恢复：高度 0.75、底座指令 32:36 为零；
躯干 RPY 仍保留模型输出。详见 [协议说明](../examples/simple_g1_dex3/README.md#0-协议一览)。

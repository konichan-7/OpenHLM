# G1 Dex3 远程推理

策略服务运行在训练环境，SIMPLE/Isaac 运行在独立的 Python 3.10 环境。
两端通过 WebSocket 交换一个第一视角 RGB、32 维状态与 `(H, 36)` 动作。

## 服务端

从 `src/openpi4OpenHLM` 运行，显式选择训练配置与 checkpoint：

```bash
PYTHONPATH=src python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_simple_g1_dex3_tabletop \
  --policy.dir checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000
```

多任务 checkpoint 对应 `pi05_simple_g1_dex3_multitask` 配置。客户端必须传入
当前任务的 prompt。服务没有其他机器人的默认 checkpoint 或 `--env` 入口。

## SIMPLE 客户端

在 SIMPLE 的独立环境中，将本仓库的客户端安装到现有 SIMPLE checkout：

```bash
bash /path/to/OpenHLM/src/openpi4OpenHLM/examples/simple_g1_dex3/install_simple_client.sh /path/to/SIMPLE
cd /path/to/SIMPLE
python -m simple.cli.eval \
  --policy pi05_openhlm \
  --env simple/G1WholebodyTabletopGraspMP-v0 \
  --host <server-ip> --port 8000
```

客户端握手校验 `action_protocol=simple_g1_dex3` 和 `action_dim=36`，发送完整
32 维状态，并在 episode reset 时清空动作队列与历史命令。源码见
[pi05_openhlm.py](../examples/simple_g1_dex3/simple_client/pi05_openhlm.py)。

服务启动后可先在训练环境执行
[`test_remote_client.py`](../examples/simple_g1_dex3/test_remote_client.py)，
使用真实数据帧验证网络协议。参数见该脚本的 `--help`。

完整训练及部署流程见 [G1 Dex3 使用说明](../examples/simple_g1_dex3/README.md)。

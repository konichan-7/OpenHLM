"""Fake-model server smoke test for ``Pi05OpenhlmAgent``'s wire contract.

Runs without a GPU or a checkpoint: it starts the real ``WebsocketPolicyServer``
with a stub policy that returns a constant 36-d chunk, connects the real
``openpi_client``, and checks the payloads the SIMPLE client will see.

Run from openpi4OpenHLM/:

    PYTHONPATH=src python examples/simple_g1_dex3/test_serve.py

This catches the things that only break over the wire: msgpack/numpy shape
handling, the metadata handshake, and the 36-d truncation assumption.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np


class StubPolicy:
    """Minimal policy with the same interface the server expects."""

    def __init__(self, action_dim: int = 36, horizon: int = 30):
        self.action_dim = action_dim
        self.horizon = horizon
        self.metadata = {"action_protocol": "simple_g1_dex3", "action_dim": action_dim, "state_dim": 32}

    def infer(self, obs: dict) -> dict:
        state = np.asarray(obs["observation/state"])
        assert state.shape == (32,), state.shape
        out = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        out[:, 31] = 0.75
        # encode the request so the client can verify it round-tripped
        image = np.asarray(obs["observation/image"])
        if image.ndim == 3:
            out[0, :3] = image.mean(axis=(0, 1))[:3]
        return {"actions": out, "policy_timing": {"infer_ms": 0.0}}


def main() -> int:
    import openpi.serving.websocket_policy_server as server_mod
    from openpi_client import websocket_client_policy

    port = 18000
    stub = StubPolicy()
    server = server_mod.WebsocketPolicyServer(policy=stub, host="127.0.0.1", port=port, metadata=stub.metadata)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(2.0)

    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port, api_key=None)
    metadata = client.get_server_metadata()
    print("metadata:", metadata)
    assert metadata["action_protocol"] == "simple_g1_dex3"
    assert int(metadata["action_dim"]) == 36

    obs = {
        "observation/image": np.zeros((360, 640, 3), dtype=np.uint8),
        "observation/state": np.zeros(32, dtype=np.float32),
        "prompt": "grasp the object on the tabletop.",
        "reset": True,
    }
    out = client.infer(obs)
    actions = np.asarray(out["actions"])
    print("actions:", actions.shape, actions.dtype)
    assert actions.shape == (30, 36), actions.shape
    assert np.isfinite(actions).all()
    assert np.allclose(actions[:, 31], 0.75)

    print("\nSERVE SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

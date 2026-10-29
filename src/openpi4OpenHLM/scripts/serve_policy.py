import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_simple_g1_dex3_tabletop").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Supply policy:checkpoint with the G1 training config and checkpoint directory.
    policy: Checkpoint | None = None

    # Number of sampling steps for action generation.
    num_steps: int = 10

    # debug mode
    debug: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Load an explicitly selected G1 Dex3 checkpoint."""
    if args.policy is None:
        raise ValueError(
            "Specify policy:checkpoint --policy.config pi05_simple_g1_dex3_tabletop "
            "--policy.dir <checkpoint-directory>"
        )
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )


def main(args: Args) -> None:
    if args.debug:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
        print("\033[91mWaiting for debugger to attach...\033[0m")
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")

    logging.info(f"Using num_steps: {args.num_steps} for action generation")
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

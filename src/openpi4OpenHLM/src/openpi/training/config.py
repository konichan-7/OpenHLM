"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.simple_g1_dex3_policy as simple_g1_dex3_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.manifests as _manifests
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to reuse statistics from a matching G1 Dex3
    training checkpoint, use:

    ```
    AssetsConfig(
        assets_dir="./checkpoints/pi05_simple_g1_dex3_tabletop/run1/3000/assets",
        asset_id="local/g1_dex3_tabletop",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional explicit episode subset (LeRobot ``episode_index`` values).  Used by
    # the SIMPLE G1/Dex3 config to train on a frozen manifest while keeping a
    # held-out validation split in the same dataset copy.  Empty means "all".
    episode_indices: tuple[int, ...] = ()

    # Optional multi-repository JSON manifest.  Each entry names a LeRobot repo
    # and its selected training episode indices.  When set, ``repo_id`` is only
    # used as the asset identifier; the loader builds a concatenated dataset from
    # the manifest instead of opening a single repository.
    dataset_manifest: str | None = None

    # Optional extracted PSI0 LeRobot v3 split.  The public 14-task release uses
    # a merged v3 parquet/video layout that is newer than this repository's
    # pinned LeRobot reader.
    psi0_v3_root: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotSimpleG1Dex3DataConfig(DataConfigFactory):
    """SIMPLE G1 + dual Dex3-1 tabletop grasp (32-d state, 36-d action).

    Uses the PSI/SIMPLE wire protocol, not the OpenHLM 34-d G1 protocol.  See
    ``openpi/policies/simple_g1_dex3_policy.py`` for the exact layout.

    ``episode_indices`` optionally restricts training and normalisation statistics
    to a fixed set of episodes (read from ``<repo_id>/meta/episodes.jsonl``
    indices).  Leave it empty to use the whole dataset.  The train/val split is
    materialised as a JSON manifest by ``examples/simple_g1_dex3/make_split.py``
    and passed in by ``scripts/train_pytorch.py --episode-manifest``.
    """

    episode_indices: tuple[int, ...] = ()
    # Optional JSON split manifest; when set, ``episode_indices`` is overridden by
    # its ``train`` list.  Exposed on the data config so ``compute_norm_stats.py``
    # (which only takes ``--config-name``) can point at a split too.
    episode_manifest: str | None = None
    # Optional multi-repository training manifest.  Kept on the data factory so
    # the training config can point at a curated set of tasks.
    dataset_manifest: str | None = None
    # Optional extracted public PSI0 v3 split (train or val).
    psi0_v3_root: str | None = None
    # Tabletop-only data has constant height/base commands; the full public
    # PSI0 dataset contains locomotion and must preserve all 36 action dims.
    pin_constant_actions: bool = True
    # Kept explicit (and False) because these actions are absolute joint targets,
    # absolute height and high-level base commands; a delta transform would change
    # their meaning.  See the plan, section 7.1.
    use_delta_joint_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        episode_indices = self.episode_indices
        if self.episode_manifest is not None:
            manifest = _manifests.load(self.episode_manifest)
            expected = manifest.get("repo_id")
            if expected and expected != self.repo_id:
                raise ValueError(f"manifest {self.episode_manifest} is for {expected!r}, config uses {self.repo_id!r}")
            episode_indices = tuple(manifest["train"])

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.egocentric",
                        "observation/state": "states",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[simple_g1_dex3_policy.SimpleG1Dex3Inputs(model_type=model_config.model_type)],
            outputs=[simple_g1_dex3_policy.SimpleG1Dex3Outputs(pin_constants=self.pin_constant_actions)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            use_quantile_norm=True,
            episode_indices=episode_indices,
            dataset_manifest=self.dataset_manifest,
            psi0_v3_root=self.psi0_v3_root,
            # The SIMPLE package names the column "action" (singular); the default
            # "actions" would make LeRobotDataset raise KeyError on delta_timestamps.
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # Initialization mode for action projection layers when action_dim > 32.
    # Pretrained models only support action_dim <= 32, so special handling is needed.
    # Options:
    #   - "weight_surgery": Preserve pretrained weights for the first 32 dimensions and
    #     initialize extra dimensions with Xavier/Glorot initialization. (Recommended)
    #   - "random": Randomly initialize all action projection layers, discarding pretrained weights.
    # This option only takes effect when pytorch_weight_path is set and action_dim > 32.
    action_proj_init_mode: Literal["weight_surgery", "random"] = "weight_surgery"

    # If True, randomly initialize action expert parameters by excluding them from checkpoint loading.
    # When enabled, the following parameter keys will be excluded from the loaded checkpoint:
    #   - Keys containing "paligemma_with_expert.gemma_expert"
    #   - Keys containing "action_in_proj"
    #   - Keys containing "action_out_proj"
    # This allows training the action expert from scratch while keeping other weights pretrained.
    random_init_action_expert: bool = False

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # If true, will use 8-bit Adam optimizer from bitsandbytes.
    use_8bit_adam: bool = False
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Optional JSON episode-split manifest (see ``openpi/training/manifests.py``).
    # When set, the data config's ``episode_indices`` is replaced by the manifest's
    # train split, both for training and for ``compute_norm_stats.py``.
    episode_manifest: str | None = None

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # How often (in steps) to run evaluation (sample actions and compute MSE with ground truth).
    # Set to 0 to disable evaluation during training.
    eval_interval: int = 20
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If true, will enable debug mode with debugpy server.
    debug: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # If true, gradient checkpointing will be enabled for memory optimization (if supported by the model).
    # This trades computation for memory by recomputing activations during the backward pass.
    enable_gradient_checkpointing: bool = True
    # If true, enable torch.compile for the PyTorch training forward path.
    enable_training_compile: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        # SIMPLE G1 + dual Dex3-1, single right-hand tabletop grasp.
        # Data:   local/g1_dex3_tabletop  (LeRobot v2.1, 32-d state, 36-d action)
        # Assets: ./assets/pi05_simple_g1_dex3_tabletop/local/g1_dex3_tabletop/norm_stats.json
        name="pi05_simple_g1_dex3_tabletop",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=30,
            discrete_state_input=True,
        ),
        data=LeRobotSimpleG1Dex3DataConfig(
            repo_id="local/g1_dex3_tabletop",
            # The dataset's only task string is "grasp the object on the tabletop.",
            # identical to the client's DEFAULT_PROMPT.  Reading it from the task
            # keeps train/serve text in sync instead of duplicating the literal.
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=3e-5,
            decay_steps=10_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
        # 3k-step first pass; extend to 5k-10k only if the validation split says so.
        num_train_steps=3_000,
        save_interval=500,
        # Full fine-tune: no LoRA, nothing frozen.
        freeze_filter=nnx.Nothing,
        # Surfaced to the client through the WebSocket handshake so a mismatched
        # server cannot silently feed a 34-d OpenHLM policy to the SIMPLE client.
        policy_metadata={
            "action_protocol": "simple_g1_dex3",
            "action_dim": 36,
            "state_dim": 32,
            "action_horizon": 30,
            "prompt": "grasp the object on the tabletop.",
        },
    ),
    TrainConfig(
        # Fixed-base, non-loco multi-task training set:
        # tabletop grasp + bend-pick MP + bend-pick teleop.
        name="pi05_simple_g1_dex3_multitask",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=30,
            discrete_state_input=True,
        ),
        data=LeRobotSimpleG1Dex3DataConfig(
            repo_id="local/g1_dex3_multitask",
            assets=AssetsConfig(asset_id="g1_dex3_multitask"),
            dataset_manifest="assets/simple_g1_dex3/multitask_manifest.json",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=4,
        num_workers=4,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=3e-5,
            decay_steps=12_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
        # One epoch over the selected non-loco training set (45k frames, batch 4).
        num_train_steps=12_000,
        save_interval=1_000,
        freeze_filter=nnx.Nothing,
        policy_metadata={
            "action_protocol": "simple_g1_dex3",
            "action_dim": 36,
            "state_dim": 32,
            "action_horizon": 30,
            "training_tasks": 3,
            "prompt": "grasp the object on the tabletop.",
        },
    ),
    TrainConfig(
        # Full public PSI0 release: 14 tasks, including locomotion and mobile
        # manipulation.  The archive's train split has 432,553 frames / 1,352
        # episodes in LeRobot v3 format.
        name="pi05_psi0_full_rtc",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=30,
            discrete_state_input=True,
            use_training_time_rtc=True,
            rtc_max_delay=10,
            rtc_mask_loss=True,
        ),
        data=LeRobotSimpleG1Dex3DataConfig(
            repo_id="local/psi0_full_v3_train",
            assets=AssetsConfig(asset_id="psi0_full_v3"),
            psi0_v3_root="/data/psi_simple_tabletop/lerobot_v3/simple_merged_v30/train",
            pin_constant_actions=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        # One pass over all 432,553 frames is about 54,070 steps at batch 8.
        batch_size=8,
        num_workers=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-5,
            decay_steps=55_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
        num_train_steps=55_000,
        save_interval=5_000,
        eval_interval=500,
        keep_period=5_000,
        freeze_filter=nnx.Nothing,
        policy_metadata={
            "action_protocol": "simple_g1_dex3",
            "action_dim": 36,
            "state_dim": 32,
            "action_horizon": 30,
            "training_tasks": 14,
            "rtc_mode": "training_time",
            "rtc_max_delay": 10,
        },
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    config = tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
    return resolve_manifest(config)


def resolve_manifest(config: TrainConfig) -> TrainConfig:
    """Apply ``config.episode_manifest`` (if any) to the data config."""
    if config.episode_manifest is None:
        return config
    manifest = _manifests.load(config.episode_manifest)
    logging.info(
        "Applying episode manifest %s: %d train / %d val episodes",
        config.episode_manifest,
        len(manifest["train"]),
        len(manifest["val"]),
    )
    return _manifests.apply_to_config(config, manifest)


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

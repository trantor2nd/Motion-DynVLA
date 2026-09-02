# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def resolve_episode_indices(entry: dict) -> list[int] | None:
    """Resolve optional include and exclude lists from a dataset manifest entry."""
    requested = entry.get("episode_indices")
    excluded = [int(index) for index in entry.get("exclude_episode_indices", [])]
    if len(excluded) != len(set(excluded)):
        raise ValueError("exclude_episode_indices contains duplicates")
    if requested is None and not excluded:
        return None

    if requested is None:
        episodes_path = Path(entry["dataset_path"]) / "meta" / "episodes.jsonl"
        with open(episodes_path, "r") as file:
            requested = [int(json.loads(line)["episode_index"]) for line in file]
    else:
        requested = [int(index) for index in requested]

    requested_set = set(requested)
    unknown_exclusions = sorted(set(excluded) - requested_set)
    if unknown_exclusions:
        raise ValueError(
            f"exclude_episode_indices not found in selected episodes: {unknown_exclusions}"
        )
    resolved = [index for index in requested if index not in set(excluded)]
    if not resolved:
        raise ValueError("episode selection is empty after exclusions")
    return resolved


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    if ft_config.dataset_manifest_path is not None:
        with open(ft_config.dataset_manifest_path, "r") as file:
            manifest = json.load(file)
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("dataset manifest must be a non-empty JSON list")
        datasets = []
        for entry in manifest:
            if not isinstance(entry, dict) or "dataset_path" not in entry:
                raise ValueError("each dataset manifest entry requires dataset_path")
            datasets.append(
                {
                    "dataset_paths": [entry["dataset_path"]],
                    "mix_ratio": float(entry.get("mix_ratio", 1.0)),
                    "embodiment_tag": entry.get("embodiment_tag", embodiment_tag),
                    "episode_indices": resolve_episode_indices(entry),
                }
            )
    else:
        datasets = [
            {
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": embodiment_tag,
            }
        ]

    config = get_default_config().load_dict(
        {"data": {"download_cache": False, "datasets": datasets}}
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_vlln = ft_config.tune_vlln
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.enable_dynvla = ft_config.enable_dynvla
    config.model.tune_dynvla = ft_config.tune_dynvla
    config.model.dynvla_constraint_depth = ft_config.dynvla_constraint_depth
    config.model.dynvla_codebook_size = ft_config.dynvla_codebook_size
    config.model.dynvla_codebook_temperature = ft_config.dynvla_codebook_temperature
    config.model.dynvla_bank_mode = ft_config.dynvla_bank_mode
    config.model.dynvla_bank_seed = ft_config.dynvla_bank_seed
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_strategy = ft_config.save_strategy
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.save_model_at_end = ft_config.save_model_at_end
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.warmup_steps = ft_config.warmup_steps
    config.training.max_grad_norm = ft_config.max_grad_norm
    config.training.adam_beta1 = ft_config.adam_beta1
    config.training.adam_beta2 = ft_config.adam_beta2
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.video_backend = ft_config.video_backend

    run(config)

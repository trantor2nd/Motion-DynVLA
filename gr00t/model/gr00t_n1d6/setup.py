import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        # Build transformers loading kwargs from training config

        if self.config.training.start_from_checkpoint is not None:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_top_llm_layers=self.config.model.tune_top_llm_layers,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                enable_dynvla=self.config.model.enable_dynvla,
                tune_dynvla=self.config.model.tune_dynvla,
                dynvla_constraint_depth=self.config.model.dynvla_constraint_depth,
                dynvla_codebook_size=self.config.model.dynvla_codebook_size,
                dynvla_codebook_temperature=self.config.model.dynvla_codebook_temperature,
                dynvla_bank_mode=self.config.model.dynvla_bank_mode,
                dynvla_bank_seed=self.config.model.dynvla_bank_seed,
                dynvla_teacher_dropout=self.config.model.dynvla_teacher_dropout,
                dynvla_mdm_hidden_dim=self.config.model.dynvla_mdm_hidden_dim,
                dynvla_mdm_layers=self.config.model.dynvla_mdm_layers,
                dynvla_mdm_dropout=self.config.model.dynvla_mdm_dropout,
                dynvla_contrastive_dim=self.config.model.dynvla_contrastive_dim,
                dynvla_contrastive_temperature=self.config.model.dynvla_contrastive_temperature,
                dynvla_ema_momentum=self.config.model.dynvla_ema_momentum,
                dynvla_lambda_temporal=self.config.model.dynvla_lambda_temporal,
                dynvla_lambda_identification=self.config.model.dynvla_lambda_identification,
                dynvla_lambda_bank_commitment=self.config.model.dynvla_lambda_bank_commitment,
                dynvla_lambda_bank_diversity=self.config.model.dynvla_lambda_bank_diversity,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                # Initialize mask_token
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            bank_key = "action_head.model.bank.prototypes"
            if self.config.model.dynvla_bank_mode != "trainable":
                if bank_key in missing_keys:
                    raise RuntimeError(
                        "Dynamics Bank was not inherited from the source checkpoint"
                    )
                unexpected_keys = loading_info.get("unexpected_keys", [])
                if bank_key in unexpected_keys:
                    raise RuntimeError(
                        "Dynamics Bank appeared as an unexpected checkpoint key"
                    )
                print(
                    "DynVLA ablation checkpoint audit: "
                    f"bank_inherited=true missing_keys={len(missing_keys)} "
                    f"unexpected_keys={len(unexpected_keys)}"
                )

            if self.config.model.dynvla_bank_mode == "random_init":
                if not model.action_head.enable_dynvla:
                    raise RuntimeError("random_init bank mode requires DynVLA")
                bank = model.action_head.model.bank
                inherited_bank = bank.prototypes.detach().clone()
                devices = [bank.prototypes.device.index] if bank.prototypes.is_cuda else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.manual_seed(self.config.model.dynvla_bank_seed)
                    bank.reset_parameters()
                if torch.equal(inherited_bank, bank.prototypes):
                    raise RuntimeError("random_init did not change Dynamics Bank prototypes")
                logging.info(
                    "Dynamics Bank was reinitialized after loading the source checkpoint "
                    f"with seed {self.config.model.dynvla_bank_seed}"
                )
                print(
                    "DynVLA bank ablation: random_init "
                    f"seed={self.config.model.dynvla_bank_seed} bank_changed=true"
                )

            # The released N1.6 checkpoint is authoritative for structural
            # state/action settings. The source dataclass defaults differ from
            # the published 3B checkpoint, so keep the processor and model in
            # lockstep after loading.
            for field_name in (
                "action_horizon",
                "max_state_dim",
                "max_action_dim",
                "apply_sincos_state_encoding",
                "use_relative_action",
            ):
                setattr(
                    self.config.model,
                    field_name,
                    getattr(model.config, field_name),
                )

        else:
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print(f"Model class: {type(model).__name__}")

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        processor_summary = {
            "max_state_dim": processor.max_state_dim,
            "max_action_dim": processor.max_action_dim,
            "max_action_horizon": processor.max_action_horizon,
            "apply_sincos_state_encoding": processor.apply_sincos_state_encoding,
            "use_relative_action": processor.use_relative_action,
        }
        print(colored(f"Processor summary: {json.dumps(processor_summary)}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)

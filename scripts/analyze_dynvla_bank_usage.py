#!/usr/bin/env python3
"""Measure Dynamics Bank retrieval usage on fixed source and target samples."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import gr00t.model  # noqa: F401  Registers the GR00T model and processor.
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType


@dataclass(frozen=True)
class SampleSpec:
    dataset_path: str
    embodiment_tag: str
    episode_index: int
    step_index: int
    source_entry_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "embodiment_tag": self.embodiment_tag,
            "episode_index": self.episode_index,
            "step_index": self.step_index,
            "source_entry_index": self.source_entry_index,
        }


class UsageAccumulator:
    def __init__(self, codebook_size: int) -> None:
        self.codebook_size = codebook_size
        self.total_rows = 0
        self.soft_mass = torch.zeros(codebook_size, dtype=torch.float64)
        self.top1_counts = torch.zeros(codebook_size, dtype=torch.int64)
        self.entropy_sum = 0.0
        self.entropy_squared_sum = 0.0
        self.finite = True

    def update(self, weights: torch.Tensor) -> None:
        probabilities = weights.detach().float().cpu().reshape(-1, weights.shape[-1])
        if probabilities.shape[-1] != self.codebook_size:
            raise ValueError(
                f"Expected {self.codebook_size} prototypes, got {probabilities.shape[-1]}"
            )
        self.finite = self.finite and bool(torch.isfinite(probabilities).all())
        if not self.finite:
            raise FloatingPointError("Non-finite Dynamics Bank weights detected")
        row_sums = probabilities.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-4, rtol=2e-4):
            raise ValueError("Dynamics Bank weights do not sum to one")

        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(
            dim=-1
        )
        self.entropy_sum += float(entropy.sum())
        self.entropy_squared_sum += float((entropy * entropy).sum())
        self.soft_mass += probabilities.double().sum(dim=0)
        top1 = probabilities.argmax(dim=-1)
        self.top1_counts += torch.bincount(top1, minlength=self.codebook_size)
        self.total_rows += probabilities.shape[0]

    def summary(self) -> dict[str, Any]:
        if self.total_rows == 0:
            raise ValueError("No Dynamics Bank retrieval rows were captured")
        mean_probability = self.soft_mass / self.total_rows
        hard_probability = self.top1_counts.double() / self.total_rows
        aggregate_entropy = float(
            -(mean_probability.clamp_min(1e-15) * mean_probability.clamp_min(1e-15).log()).sum()
        )
        entropy_mean = self.entropy_sum / self.total_rows
        entropy_variance = max(
            0.0,
            self.entropy_squared_sum / self.total_rows - entropy_mean * entropy_mean,
        )
        log_codebook = math.log(self.codebook_size)
        used = int((self.top1_counts > 0).sum())
        soft_sorted, soft_indices = torch.sort(mean_probability, descending=True)
        hard_sorted, hard_indices = torch.sort(hard_probability, descending=True)
        effective_count = math.exp(aggregate_entropy)
        hard_entropy = float(
            -(hard_probability.clamp_min(1e-15) * hard_probability.clamp_min(1e-15).log()).sum()
        )
        return {
            "finite": self.finite,
            "retrieval_rows": self.total_rows,
            "top1_used_prototypes": used,
            "top1_unused_prototypes": self.codebook_size - used,
            "top1_usage_rate": used / self.codebook_size,
            "effective_prototype_count_soft": effective_count,
            "effective_prototype_fraction_soft": effective_count / self.codebook_size,
            "mean_individual_entropy": entropy_mean,
            "std_individual_entropy": math.sqrt(entropy_variance),
            "mean_individual_entropy_normalized": entropy_mean / log_codebook,
            "aggregate_soft_entropy": aggregate_entropy,
            "aggregate_soft_entropy_normalized": aggregate_entropy / log_codebook,
            "aggregate_hard_entropy": hard_entropy,
            "aggregate_hard_entropy_normalized": hard_entropy / log_codebook,
            "soft_max_mass": float(soft_sorted[0]),
            "soft_top10_mass": float(soft_sorted[:10].sum()),
            "soft_hhi": float((mean_probability * mean_probability).sum()),
            "hard_max_share": float(hard_sorted[0]),
            "hard_top10_share": float(hard_sorted[:10].sum()),
            "hard_hhi": float((hard_probability * hard_probability).sum()),
            "head_soft_prototypes": [
                {"prototype": int(index), "mass": float(mass)}
                for mass, index in zip(soft_sorted[:10], soft_indices[:10], strict=True)
            ],
            "head_hard_prototypes": [
                {"prototype": int(index), "share": float(share)}
                for share, index in zip(hard_sorted[:10], hard_indices[:10], strict=True)
            ],
            "monopoly_flags": {
                "single_top1_over_10_percent": bool(hard_sorted[0] > 0.10),
                "top10_over_50_percent": bool(hard_sorted[:10].sum() > 0.50),
                "effective_fraction_below_10_percent": bool(
                    effective_count / self.codebook_size < 0.10
                ),
            },
        }


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_records(dataset_path: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    with open(episodes_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[int(record["episode_index"])] = record
    if not records:
        raise ValueError(f"No episodes found in {episodes_path}")
    return records


def resolved_episode_indices(entry: dict[str, Any], records: dict[int, dict[str, Any]]) -> list[int]:
    requested = entry.get("episode_indices")
    indices = sorted(records) if requested is None else [int(value) for value in requested]
    excluded = {int(value) for value in entry.get("exclude_episode_indices", [])}
    unknown = sorted((set(indices) | excluded) - set(records))
    if unknown:
        raise ValueError(f"Unknown episode indices for {entry['dataset_path']}: {unknown}")
    indices = [value for value in indices if value not in excluded]
    if not indices:
        raise ValueError(f"Empty episode selection for {entry['dataset_path']}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate episode selection for {entry['dataset_path']}")
    return indices


def build_sample_plan(
    manifest_path: str,
    samples_per_entry: int,
    seed: int,
    action_delta_indices: list[int],
) -> list[SampleSpec]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Manifest must be a non-empty list: {manifest_path}")
    rng = np.random.default_rng(seed)
    lower_offset = min(action_delta_indices)
    upper_offset = max(action_delta_indices)
    plan: list[SampleSpec] = []
    for entry_index, entry in enumerate(manifest):
        records = episode_records(entry["dataset_path"])
        candidates = resolved_episode_indices(entry, records)
        count = min(samples_per_entry, len(candidates))
        selected = rng.choice(candidates, size=count, replace=False).tolist()
        for episode_index in selected:
            length = int(records[int(episode_index)]["length"])
            first_step = max(0, -lower_offset)
            last_step = min(length - 1, length - 1 - upper_offset)
            if last_step < first_step:
                raise ValueError(
                    f"Episode {episode_index} in {entry['dataset_path']} is too short"
                )
            step_index = int(rng.integers(first_step, last_step + 1))
            plan.append(
                SampleSpec(
                    dataset_path=entry["dataset_path"],
                    embodiment_tag=entry["embodiment_tag"],
                    episode_index=int(episode_index),
                    step_index=step_index,
                    source_entry_index=entry_index,
                )
            )
    if not plan:
        raise ValueError(f"No samples selected from {manifest_path}")
    return plan


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def process_samples(
    plan: list[SampleSpec],
    processor: Any,
) -> Iterable[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SampleSpec]] = defaultdict(list)
    for spec in plan:
        grouped[(spec.dataset_path, spec.embodiment_tag)].append(spec)
    modality_configs = processor.get_modality_configs()
    for (dataset_path, embodiment_value), specs in grouped.items():
        embodiment = EmbodimentTag(embodiment_value)
        loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs[embodiment.value],
            video_backend="pyav",
            episode_indices=[spec.episode_index for spec in specs],
        )
        loader_index_by_episode = {
            int(metadata["episode_index"]): index
            for index, metadata in enumerate(loader.episodes_metadata)
        }
        for spec in specs:
            episode_data = loader[loader_index_by_episode[spec.episode_index]]
            step_data = extract_step_data(
                episode_data=episode_data,
                step_index=spec.step_index,
                modality_configs=modality_configs[embodiment.value],
                embodiment_tag=embodiment,
                allow_padding=False,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            yield processor(messages)
            del episode_data, step_data, messages


def checkpoint_mode(checkpoint: str) -> str:
    config = read_json(Path(checkpoint) / "config.json")
    return str(config.get("dynvla_bank_mode", "trainable"))


def evaluate_checkpoint(
    label: str,
    checkpoint: str,
    plans: dict[str, list[SampleSpec]],
    datasets_to_run: list[str],
    batch_size: int,
    seed: int,
    device: str,
    expected_codebook_size: int,
) -> dict[str, Any]:
    real_checkpoint = os.path.realpath(checkpoint)
    mode = checkpoint_mode(real_checkpoint)
    if mode == "disabled" or label == "disabled":
        return {
            "label": label,
            "checkpoint": real_checkpoint,
            "bank_mode": mode,
            "available": False,
            "reason": "Dynamics Bank retrieval is bypassed by design in disabled mode",
        }

    processor = AutoProcessor.from_pretrained(real_checkpoint, local_files_only=True)
    processor.eval()
    model = AutoModel.from_pretrained(
        real_checkpoint,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    model.action_head.beta_dist = torch.distributions.Beta(
        torch.tensor(float(model.config.noise_beta_alpha), dtype=torch.float32),
        torch.tensor(float(model.config.noise_beta_beta), dtype=torch.float32),
    )
    bank = model.action_head.model.bank
    prototypes = bank.prototypes.detach()
    if prototypes.ndim != 2 or prototypes.shape[0] != expected_codebook_size:
        raise ValueError(f"Unexpected prototype shape: {tuple(prototypes.shape)}")
    if not torch.isfinite(prototypes.float()).all():
        raise FloatingPointError("Non-finite prototype tensor detected")

    output: dict[str, Any] = {
        "label": label,
        "checkpoint": real_checkpoint,
        "bank_mode": mode,
        "available": True,
        "prototype_shape": list(prototypes.shape),
        "prototype_finite": True,
        "datasets": {},
    }
    for dataset_name in datasets_to_run:
        accumulators = {
            "identified_query": UsageAccumulator(expected_codebook_size),
            "teacher_target": UsageAccumulator(expected_codebook_size),
        }

        def capture(_module: Any, inputs: tuple[Any, ...], result: tuple[Any, Any]) -> None:
            queries = inputs[0]
            weights = result[1]
            branch = "teacher_target" if queries.shape[1] == 1 else "identified_query"
            accumulators[branch].update(weights)

        handle = bank.register_forward_hook(capture)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        processed = process_samples(plans[dataset_name], processor)
        sample_count = 0
        try:
            pending: list[dict[str, Any]] = []
            for sample in processed:
                pending.append(sample)
                if len(pending) < batch_size:
                    continue
                collated = processor.collator(pending)
                with torch.inference_mode():
                    model(**collated)
                sample_count += len(pending)
                pending.clear()
            if pending:
                collated = processor.collator(pending)
                with torch.inference_mode():
                    model(**collated)
                sample_count += len(pending)
                pending.clear()
        finally:
            handle.remove()
        if sample_count != len(plans[dataset_name]):
            raise RuntimeError(
                f"Processed {sample_count} samples, expected {len(plans[dataset_name])}"
            )
        output["datasets"][dataset_name] = {
            "samples": sample_count,
            "identified_query": accumulators["identified_query"].summary(),
            "teacher_target": accumulators["teacher_target"].summary(),
        }

    del model, processor, bank, prototypes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    keys = [
        "top1_usage_rate",
        "effective_prototype_count_soft",
        "mean_individual_entropy_normalized",
        "aggregate_soft_entropy_normalized",
        "soft_max_mass",
        "soft_top10_mass",
        "hard_max_share",
        "hard_top10_share",
    ]
    return {key: float(right[key] - left[key]) for key in keys}


def build_comparisons(checkpoints: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for dataset_name in ("source", "target"):
        if not all(
            label in checkpoints
            and checkpoints[label].get("available")
            and dataset_name in checkpoints[label].get("datasets", {})
            for label in ("source", "target")
        ):
            continue
        comparisons[f"source_to_target_on_{dataset_name}"] = {}
        for branch in ("identified_query", "teacher_target"):
            comparisons[f"source_to_target_on_{dataset_name}"][branch] = metric_delta(
                checkpoints["source"]["datasets"][dataset_name][branch],
                checkpoints["target"]["datasets"][dataset_name][branch],
            )
    for ablation in ("frozen", "random_init"):
        if not all(
            label in checkpoints
            and checkpoints[label].get("available")
            and "target" in checkpoints[label].get("datasets", {})
            for label in ("target", ablation)
        ):
            continue
        comparisons[f"target_to_{ablation}_on_target"] = {}
        for branch in ("identified_query", "teacher_target"):
            comparisons[f"target_to_{ablation}_on_target"][branch] = metric_delta(
                checkpoints["target"]["datasets"]["target"][branch],
                checkpoints[ablation]["datasets"]["target"][branch],
            )
    return comparisons


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_assignment)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--source-samples-per-entry", type=int, default=4)
    parser.add_argument("--target-samples-per-entry", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-codebook-size", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.source_samples_per_entry <= 0 or args.target_samples_per_entry <= 0:
        raise ValueError("Samples per entry must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("Checkpoint labels must be unique")
    for label, path in checkpoints.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f"Checkpoint not found for {label}: {path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    planning_processor = AutoProcessor.from_pretrained(
        os.path.realpath(next(iter(checkpoints.values()))), local_files_only=True
    )
    modality_configs = planning_processor.get_modality_configs()
    source_manifest = read_json(args.source_manifest)
    target_manifest = read_json(args.target_manifest)
    first_embodiment = EmbodimentTag(source_manifest[0]["embodiment_tag"])
    source_action_indices = modality_configs[first_embodiment.value]["action"].delta_indices
    target_embodiment = EmbodimentTag(target_manifest[0]["embodiment_tag"])
    target_action_indices = modality_configs[target_embodiment.value]["action"].delta_indices
    plans = {
        "source": build_sample_plan(
            args.source_manifest,
            args.source_samples_per_entry,
            args.seed,
            source_action_indices,
        ),
        "target": build_sample_plan(
            args.target_manifest,
            args.target_samples_per_entry,
            args.seed + 1,
            target_action_indices,
        ),
    }
    del planning_processor

    results: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        datasets_to_run = ["source", "target"] if label in {"source", "target"} else ["target"]
        results[label] = evaluate_checkpoint(
            label=label,
            checkpoint=checkpoint,
            plans=plans,
            datasets_to_run=datasets_to_run,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            expected_codebook_size=args.expected_codebook_size,
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "diagnostic": "Dynamics Bank retrieval usage",
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "expected_codebook_size": args.expected_codebook_size,
        "torch_version": torch.__version__,
        "flash_attention_version": importlib.metadata.version("flash_attn"),
        "weight_saving": False,
        "sampling": {
            "source": {
                "manifest": os.path.realpath(args.source_manifest),
                "manifest_sha256": sha256_file(args.source_manifest),
                "samples_per_entry": args.source_samples_per_entry,
                "sample_count": len(plans["source"]),
                "samples": [spec.as_dict() for spec in plans["source"]],
            },
            "target": {
                "manifest": os.path.realpath(args.target_manifest),
                "manifest_sha256": sha256_file(args.target_manifest),
                "samples_per_entry": args.target_samples_per_entry,
                "sample_count": len(plans["target"]),
                "samples": [spec.as_dict() for spec in plans["target"]],
            },
        },
        "definitions": {
            "unused_prototype": "Prototype never selected by top1 retrieval in sampled rows",
            "effective_prototype_count_soft": "exp entropy of aggregate soft retrieval mass",
            "identified_query": "Retrieval from the 16 midpoint layer queries",
            "teacher_target": "Retrieval from the ground truth trajectory dynamics encoder",
            "monopoly_thresholds": {
                "single_top1_share": 0.10,
                "top10_share": 0.50,
                "effective_fraction": 0.10,
            },
        },
        "checkpoints": results,
        "comparisons": build_comparisons(results),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": os.path.realpath(args.output),
        "source_samples": len(plans["source"]),
        "target_samples": len(plans["target"]),
        "checkpoints": list(results),
    }, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open


PROTOTYPE_KEY = "action_head.model.bank.prototypes"


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path).resolve()


def parse_pair(value: str) -> tuple[str, str]:
    parts = value.split(",", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("expected LABEL_A,LABEL_B")
    return parts[0], parts[1]


def quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values, probability).item())


def load_prototypes(checkpoint: Path) -> tuple[torch.Tensor, Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing model index at {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map") or {}
    shard_name = weight_map.get(PROTOTYPE_KEY)
    if not shard_name:
        raise RuntimeError(f"missing {PROTOTYPE_KEY} in {index_path}")
    shard_path = checkpoint / shard_name
    if not shard_path.is_file():
        raise RuntimeError(f"missing prototype shard at {shard_path}")
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        if PROTOTYPE_KEY not in handle.keys():
            raise RuntimeError(f"prototype key is absent from {shard_path}")
        tensor = handle.get_tensor(PROTOTYPE_KEY).float()
    if tensor.ndim != 2 or tensor.shape[0] != 1024:
        raise RuntimeError(f"unexpected prototype shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError("prototype tensor contains non-finite values")
    return tensor, shard_path


def checkpoint_statistics(tensor: torch.Tensor) -> dict:
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    normalized = torch.nn.functional.normalize(tensor, dim=-1)
    cosine = normalized @ normalized.T
    diagonal_mask = torch.eye(cosine.shape[0], dtype=torch.bool)
    off_diagonal = cosine[~diagonal_mask]
    nearest = cosine.masked_fill(diagonal_mask, -torch.inf).amax(dim=-1)
    return {
        "shape": list(tensor.shape),
        "dtype_after_load": str(tensor.dtype),
        "finite": True,
        "exact_unique_rows": int(torch.unique(tensor, dim=0).shape[0]),
        "norm": {
            "mean": float(norms.mean().item()),
            "std": float(norms.std(unbiased=False).item()),
            "min": float(norms.min().item()),
            "median": quantile(norms, 0.5),
            "max": float(norms.max().item()),
        },
        "off_diagonal_cosine": {
            "mean": float(off_diagonal.mean().item()),
            "std": float(off_diagonal.std(unbiased=False).item()),
            "p95": quantile(off_diagonal, 0.95),
            "p99": quantile(off_diagonal, 0.99),
            "max": float(off_diagonal.max().item()),
        },
        "nearest_neighbor_cosine": {
            "mean": float(nearest.mean().item()),
            "median": quantile(nearest, 0.5),
            "p95": quantile(nearest, 0.95),
            "max": float(nearest.max().item()),
        },
    }


def comparison_statistics(source: torch.Tensor, target: torch.Tensor) -> dict:
    if source.shape != target.shape:
        raise RuntimeError(
            f"cannot compare prototype shapes {tuple(source.shape)} and {tuple(target.shape)}"
        )
    delta = target - source
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(source, target, dim=-1)
    source_frobenius = torch.linalg.vector_norm(source)
    delta_frobenius = torch.linalg.vector_norm(delta)
    return {
        "same_shape": True,
        "exact_equal": bool(torch.equal(source, target)),
        "unchanged_rows_at_1e_7": int((delta_norm <= 1e-7).sum().item()),
        "relative_frobenius_delta": float(
            (delta_frobenius / source_frobenius.clamp_min(1e-12)).item()
        ),
        "row_l2_delta": {
            "mean": float(delta_norm.mean().item()),
            "median": quantile(delta_norm, 0.5),
            "p95": quantile(delta_norm, 0.95),
            "max": float(delta_norm.max().item()),
        },
        "same_index_cosine": {
            "mean": float(cosine.mean().item()),
            "median": quantile(cosine, 0.5),
            "p05": quantile(cosine, 0.05),
            "min": float(cosine.min().item()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--compare",
        action="append",
        type=parse_pair,
        default=[],
        metavar="LABEL_A,LABEL_B",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_paths = dict(args.checkpoint)
    if len(checkpoint_paths) != len(args.checkpoint):
        raise RuntimeError("checkpoint labels must be unique")
    tensors: dict[str, torch.Tensor] = {}
    checkpoints: dict[str, dict] = {}
    for label, checkpoint in checkpoint_paths.items():
        tensor, shard = load_prototypes(checkpoint)
        tensors[label] = tensor
        checkpoints[label] = {
            "checkpoint": str(checkpoint),
            "parameter_key": PROTOTYPE_KEY,
            "shard": str(shard),
            "statistics": checkpoint_statistics(tensor),
        }

    comparisons = {}
    for source_label, target_label in args.compare:
        if source_label not in tensors or target_label not in tensors:
            raise RuntimeError(
                f"unknown comparison labels {source_label},{target_label}"
            )
        pair_label = f"{source_label}_to_{target_label}"
        if pair_label in comparisons:
            raise RuntimeError(f"duplicate comparison {pair_label}")
        comparisons[pair_label] = comparison_statistics(
            tensors[source_label], tensors[target_label]
        )

    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "prototype_key": PROTOTYPE_KEY,
        "checkpoints": checkpoints,
        "comparisons": comparisons,
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(encoded)
    os.replace(temporary_path, output_path)
    print(encoded, end="")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()

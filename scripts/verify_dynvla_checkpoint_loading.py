"""Verify that DynVLA adds parameters without losing official N1.6 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel

import gr00t.model  # noqa: F401


NEW_PARAMETER_PREFIXES = (
    "action_head.model.identification_token",
    "action_head.model.query_projection.",
    "action_head.model.bank.",
    "action_head.trajectory_dynamics_encoder.",
    "action_head.temporal_grounding.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.resolve()

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as file:
        official_keys = set(json.load(file)["weight_map"])

    model, loading_info = AutoModel.from_pretrained(
        model_path,
        enable_dynvla=True,
        tune_llm=False,
        tune_top_llm_layers=0,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        tune_dynvla=True,
        local_files_only=True,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    model_keys = set(model.state_dict())
    structurally_missing = sorted(official_keys - model_keys)
    if structurally_missing:
        raise RuntimeError(f"official parameter keys were renamed or removed: {structurally_missing}")

    unexpected = sorted(loading_info.get("unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"unexpected official checkpoint keys: {unexpected}")

    missing = sorted(loading_info.get("missing_keys", []))
    disallowed_missing = [
        key for key in missing if not key.startswith(NEW_PARAMETER_PREFIXES)
    ]
    if disallowed_missing:
        raise RuntimeError(f"non-DynVLA parameters were not loaded: {disallowed_missing}")

    nonfinite_new_parameters = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(NEW_PARAMETER_PREFIXES)
        and not torch.isfinite(parameter.detach()).all()
    ]
    if nonfinite_new_parameters:
        raise RuntimeError(
            f"new DynVLA parameters were not initialized: {nonfinite_new_parameters}"
        )

    print(f"official_checkpoint_keys={len(official_keys)}")
    print(f"dynvla_model_keys={len(model_keys)}")
    print(f"new_dynvla_keys={len(missing)}")
    print("new_dynvla_parameters_finite=true")
    print("official_parameter_coverage=100_percent")


if __name__ == "__main__":
    main()

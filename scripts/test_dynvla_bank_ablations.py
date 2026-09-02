#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


TEST_NAMES = (
    "test_dynvla_preserves_pretrained_dit_parameter_names",
    "test_dynvla_midpoint_retrieval_has_gradients",
    "test_dynvla_frozen_bank_keeps_other_dynamics_trainable",
    "test_dynvla_random_bank_reset_changes_only_prototypes",
    "test_dynvla_disabled_bank_bypasses_retrieval_and_stays_finite",
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(
        str(repository_root / "tests" / "gr00t" / "model" / "test_dynvla_n1d6.py")
    )
    for name in TEST_NAMES:
        namespace[name]()
        print(f"PASS {name}")
    print(f"PASS total={len(TEST_NAMES)}")


if __name__ == "__main__":
    main()

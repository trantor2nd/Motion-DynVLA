"""Verify the pinned GR00T N1.6 runtime and official Flash Attention wheel."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import flash_attn
import torch
import transformers


EXPECTED = {
    "flash-attn": "2.7.4.post1",
    "torch": "2.7.1",
    "transformers": "4.51.3",
}


def main() -> None:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        if actual[name] != expected:
            raise RuntimeError(f"{name} version {actual[name]} does not match {expected}")

    flash_path = Path(flash_attn.__file__).resolve()
    if "site-packages" not in flash_path.parts:
        raise RuntimeError(f"unexpected Flash Attention import path {flash_path}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"PyTorch CUDA version is {torch.version.cuda}, expected 12.8")

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"transformers={transformers.__version__}")
    print(f"flash_attn={actual['flash-attn']}")
    print(f"flash_attn_path={flash_path}")


if __name__ == "__main__":
    main()

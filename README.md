# Motion-DynVLA

Motion Dynamics Learning for Few-Shot Embodied Adaptation (ICML 2026 Poster)

[![Hugging Face weights](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Weights-FFD21E)](https://huggingface.co/trantor2nd/Motion-DynVLA)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?logo=adobeacrobatreader&logoColor=white)](paper/Motion-DynVLA.pdf)

Motion-DynVLA equips a pretrained vision-language-action model with an explicit motion dynamics module and a reusable Dynamics Bank. The model first learns motion structure from source-domain trajectories and then adapts to a new task from only 20 demonstrations.

## Highlights

- Built on the full NVIDIA GR00T N1.6 pretrained VLA and action expert
- Adds midpoint motion identification without replacing pretrained action layers
- Uses a learnable bank of 1024 motion prototypes
- Supports source-domain pretraining followed by fixed 20-demonstration adaptation
- Includes fixed LIBERO data manifests, training entry points and paired evaluation tools
- Includes frozen, random initialization and disabled-bank ablations
- Includes auditable prototype drift and retrieval usage diagnostics

## Method

Motion-DynVLA inserts a motion identification token into the GR00T diffusion transformer. The first 16 DiT layers produce motion queries. A differentiable Dynamics Bank retrieves reusable motion prototypes and injects them into the remaining DiT layers.

Training follows three stages.

1. Source dynamics warmup for the trajectory dynamics encoder and Dynamics Bank
2. Source joint training for the pretrained action expert and dynamics modules
3. Target adaptation from exactly 20 demonstrations per task

The visual-language backbone remains frozen. The pretrained action encoder, DiT and action decoder remain the initialization for both source training and few-shot adaptation.

## Validated results

| Setting | Result |
| --- | --- |
| `libero-target20` on LIBERO Long | 428 successes in 500 fixed episodes (85.6%) |
| LIBERO gain from the pretrained bank over random initialization | 3.6 percentage points |

All comparisons use paired initial states or paired seeds. The repository contains deterministic manifests and resumable evaluators for reproducing the protocol.

## Repository layout

```text
configs/data_manifests       Fixed source and target20 data selections
examples/LIBERO              LIBERO modality configuration
gr00t/model/modules           Motion dynamics and Dynamics Bank modules
scripts/run_dynvla_stage.sh   Three-stage training entry point
scripts/eval                  LIBERO evaluation tools
scripts/analyze_dynvla_bank_prototypes.py
scripts/analyze_dynvla_bank_usage.py
tests/gr00t/model/test_dynvla_n1d6.py
paper/Motion-DynVLA.pdf    Paper manuscript
```

## Environment

The validated environment uses the following core versions.

```text
Python             3.10
PyTorch            2.7.1
CUDA               12.8
Transformers       4.51.3
Flash Attention    2.7.4.post1
```

Flash Attention is enabled. The project lock file installs the official prebuilt wheel and does not compile Flash Attention online.

For a fresh checkout, the commands below create the directory layout used by the launchers on Linux.

```bash
export MOTION_DYNVLA_ROOT="${MOTION_DYNVLA_ROOT:-$HOME/motion_dynvla}"
mkdir -p "$MOTION_DYNVLA_ROOT/code" "$MOTION_DYNVLA_ROOT/envs"
git clone https://github.com/trantor2nd/Motion-DynVLA.git "$MOTION_DYNVLA_ROOT/code/DynVLA-GR00T"
cd "$MOTION_DYNVLA_ROOT/code/DynVLA-GR00T"
export UV_PROJECT_ENVIRONMENT="$MOTION_DYNVLA_ROOT/envs/gr00t_n16"
uv sync --frozen --python 3.10 --no-build-package flash-attn
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/verify_dynvla_gr00t_n16_env.py
```

The benchmark simulators and converted LeRobot datasets should remain outside the Git repository under the project root.

## Model weights

Both downloads contain the full DynVLA policy, including the GR00T action expert and the Dynamics Bank. Choose the checkpoint for the stage you want to run.

| Checkpoint | How it was trained | When to use it |
| --- | --- | --- |
| [`libero-source`](https://huggingface.co/trantor2nd/Motion-DynVLA/tree/main/libero-source) | Starting from NVIDIA GR00T N1.6, the dynamics modules were warmed up for 2,000 steps on LIBERO Spatial, Object and Goal (30 source tasks). The action expert and dynamics modules then trained jointly on the same tasks for 28,000 steps. | Start a new adaptation run with the learned Dynamics Bank, or evaluate the policy before target adaptation. |
| [`libero-target20`](https://huggingface.co/trantor2nd/Motion-DynVLA/tree/main/libero-target20) | Starting from `libero-source`, the policy was fine-tuned for 30,000 steps on LIBERO Long (`libero_10`). Each of the 10 target tasks contributed 20 fixed demonstrations, for 200 in total. | Reproduce the reported 85.6% success rate on LIBERO Long. |

The 200 target demonstrations enter our training pipeline only during target adaptation. See the [source task manifest](configs/data_manifests/libero_source30.json) and the [fixed target episode list](configs/data_manifests/libero_target10_20shot.json).

Download the evaluation checkpoint with `huggingface_hub` in the configured project environment.

```python
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="trantor2nd/Motion-DynVLA",
    allow_patterns=["libero-target20/*"],
    local_dir=Path(os.environ["MOTION_DYNVLA_ROOT"]) / "checkpoints" / "Motion-DynVLA",
)
```

Each published checkpoint contains the model shards, model index, processor configuration, embodiment mapping and normalization statistics. Optimizer and scheduler states are intentionally excluded from the public inference package.

## Training

Use the environment above and prepare the [LIBERO datasets](examples/LIBERO/README.md). Place the NVIDIA base checkpoint under `$MOTION_DYNVLA_ROOT/models/GR00T-N1.6-3B`. Run the three stages in sequence to train both released policies.

```bash
bash scripts/run_dynvla_stage.sh libero source_warmup \
  "$MOTION_DYNVLA_ROOT/models/GR00T-N1.6-3B"

bash scripts/run_dynvla_stage.sh libero source_joint \
  "$MOTION_DYNVLA_ROOT/runs/gr00t_n16/dynvla_gr00t_n16_libero_source_warmup_final"

bash scripts/run_dynvla_stage.sh libero target20 \
  "$MOTION_DYNVLA_ROOT/runs/gr00t_n16/dynvla_gr00t_n16_libero_source_joint_final"
```

Each formal run uses one GPU. Independent runs can be assigned to different GPUs without cross-GPU synchronization.

## Evaluation

With the LIBERO simulator installed under `$MOTION_DYNVLA_ROOT/third_party/LIBERO` and its Python environment at `$MOTION_DYNVLA_ROOT/envs/libero`, evaluate the downloaded `libero-target20` policy on all 10 tasks with 50 initial states per task.

```bash
bash scripts/eval/run_libero_gr00t.sh \
  "$MOTION_DYNVLA_ROOT/checkpoints/Motion-DynVLA/libero-target20" \
  0 5562 0 10 50 0 libero_target20
```

## Dynamics Bank diagnostics

The repository provides two no-weight-save diagnostics.

```bash
python scripts/analyze_dynvla_bank_prototypes.py --help
python scripts/analyze_dynvla_bank_usage.py --help
```

They report prototype drift, retrieval entropy, effective prototype count, top1 usage, unused prototypes and concentration statistics.

## Acknowledgements

This project builds on [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [GR00T N1.6](https://huggingface.co/nvidia/GR00T-N1.6-3B) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## License

The repository follows the included NVIDIA License and its non-commercial research use terms. Third-party components and datasets remain subject to their respective licenses.

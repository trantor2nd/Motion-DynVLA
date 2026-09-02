# Motion-DynVLA

Motion Dynamics Learning for Few-Shot Embodied Adaptation on NVIDIA GR00T N1.6.

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
| LIBERO with 20 demonstrations per task | 85.6 percent success over 500 fixed episodes |
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

Prepare a project root and place this repository under `code/DynVLA-GR00T`.

```bash
export MOTION_DYNVLA_ROOT=/path/to/motion_dynvla
cd "$MOTION_DYNVLA_ROOT/code/DynVLA-GR00T"
uv sync --frozen --no-build-package flash-attn
python scripts/verify_dynvla_gr00t_n16_env.py
```

The benchmark simulators and converted LeRobot datasets should remain outside the Git repository under the project root.

## Model weights

The public Hugging Face repository provides two inference-ready checkpoints.

```text
libero-source       Source-domain LIBERO checkpoint at step 28000
libero-target20     LIBERO checkpoint adapted with 20 demonstrations per task
```

Download one checkpoint with `huggingface_hub`.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="trantor2nd/Motion-DynVLA",
    allow_patterns=["libero-target20/*"],
    local_dir="./checkpoints/Motion-DynVLA",
)
```

Each published checkpoint contains the model shards, model index, processor configuration, embodiment mapping and normalization statistics. Optimizer and scheduler states are intentionally excluded from the public inference package.

## Training

Set the project root before launching a stage.

```bash
export MOTION_DYNVLA_ROOT=/path/to/motion_dynvla
```

Train the LIBERO source model and then adapt it with 20 demonstrations.

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

Evaluate a LIBERO checkpoint.

```bash
bash scripts/eval/run_libero_gr00t.sh \
  "$MOTION_DYNVLA_ROOT/runs/gr00t_n16/dynvla_gr00t_n16_libero_target20_final" \
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

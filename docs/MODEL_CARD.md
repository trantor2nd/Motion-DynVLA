# Motion-DynVLA

Motion-DynVLA is a motion-dynamics extension of NVIDIA GR00T N1.6 for few-shot embodied adaptation. It retains the pretrained vision-language-action backbone and action expert, then adds midpoint motion identification and a reusable bank of 1024 motion prototypes.

## Checkpoints

| Folder | Description |
| --- | --- |
| `libero-source` | LIBERO source-domain checkpoint after dynamics warmup and joint training |
| `libero-target20` | LIBERO checkpoint adapted with 20 demonstrations per target task |

Each folder is directly loadable with the Motion-DynVLA code through `AutoModel.from_pretrained` and `AutoProcessor.from_pretrained`.

## Validated results

| Setting | Result |
| --- | --- |
| LIBERO with 20 demonstrations per task | 85.6 percent success over 500 fixed episodes |
| LIBERO gain from the pretrained bank over random initialization | 3.6 percentage points |

The public checkpoints contain inference weights and processor assets. Optimizer, scheduler, RNG and trainer state files are excluded.

## Loading

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="trantor2nd/Motion-DynVLA",
    allow_patterns=["libero-target20/*"],
    local_dir="./checkpoints/Motion-DynVLA",
)
```

## Code

The implementation, fixed manifests and evaluation tools are available in [Motion-DynVLA](https://github.com/trantor2nd/Motion-DynVLA).

## Base model and benchmarks

Motion-DynVLA builds on [NVIDIA GR00T N1.6](https://huggingface.co/nvidia/GR00T-N1.6-3B). The published checkpoints are validated with paired evaluation on LIBERO.

## License

These checkpoints are released under the NVIDIA License included with the code repository. Use is limited to non-commercial research under those terms. Third-party components remain subject to their respective licenses.

"""Map the existing converted LIBERO camera names into the N1.6 config."""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag


libero_config = deepcopy(MODALITY_CONFIGS[EmbodimentTag.LIBERO_PANDA.value])
libero_config["video"].modality_keys = ["primary_image", "wrist_image"]
MODALITY_CONFIGS[EmbodimentTag.LIBERO_PANDA.value] = libero_config

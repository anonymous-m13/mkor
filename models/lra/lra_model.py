from .lra_config import config
import math
from .model_wrapper import ModelForSCDual, ModelForSC

__all__ = ['lra_model']


def lra_model(task, **kwargs):
    attn_type = "softmax"
    model_config = config[task]["model"]
    model_config.update(config[task]["extra_attn_config"][attn_type])

    model_config["mixed_precision"] = False
    model_config["attn_type"] = attn_type
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    if task == "retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)
    return model
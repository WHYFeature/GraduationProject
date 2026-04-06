from dataclasses import dataclass
from typing import List, Optional, Tuple

from memit_project.utils.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: Optional[float]

    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    batch_size: int = 64
    wd_power_law: Optional[Tuple[float, float]] = None

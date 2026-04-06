from dataclasses import dataclass
from typing import List, Optional

from memit_project.utils.hparams import HyperParams


@dataclass
class MENDHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    zsre: bool
    mini: bool
    checkpoint_path: Optional[str] = None
    layers: Optional[List[int]] = None

    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

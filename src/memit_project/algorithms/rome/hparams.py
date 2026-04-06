from dataclasses import dataclass
from typing import List, Literal, Tuple

from memit_project.utils.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    layers: List[int]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[Tuple[int, int]]

    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

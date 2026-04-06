from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.utils import nethook
from memit_project.utils.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    **_: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        deltas = execute_rome(
            model, tok, request, hparams, (cache_template if i == 0 else None)
        )
        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()
                w[...] += upd_matrix

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    deltas = {}
    hparams.layers = sorted(hparams.layers)

    for layer in hparams.layers:
        left_vector, right_vector = None, None
        require_recompute = True
        cache_fname = None

        if layer == hparams.layers[0]:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            if cache_fname is not None and cache_fname.exists():
                try:
                    data = np.load(cache_fname)
                    left_vector = torch.from_numpy(data["left_vector"]).to("cuda")
                    right_vector = torch.from_numpy(data["right_vector"]).to("cuda")
                    require_recompute = False
                except Exception:
                    require_recompute = True

        left_vector = (
            left_vector
            if left_vector is not None
            else compute_u(
                model,
                tok,
                request,
                hparams,
                layer,
                get_context_templates(
                    model, tok, hparams.context_template_length_params
                ),
            )
        )
        right_vector = (
            right_vector
            if right_vector is not None
            else compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(
                    model, tok, hparams.context_template_length_params
                ),
            )
        )

        if cache_fname is not None and require_recompute:
            cache_fname.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                cache_fname,
                left_vector=left_vector.detach().cpu().numpy(),
                right_vector=right_vector.detach().cpu().numpy(),
            )

        with torch.no_grad():
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (left_vector.detach(), right_vector.detach())

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape."
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
    return CONTEXT_TEMPLATES_CACHE

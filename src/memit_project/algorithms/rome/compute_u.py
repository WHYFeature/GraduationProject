from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.algorithms.memit.editor import get_cov

from .hparams import ROMEHyperParams
from . import repr_tools


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    u = cur_repr
    if hparams.mom2_adjustment:
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ).double()
        u = torch.linalg.solve(cov, u.double().unsqueeze(1)).squeeze(1).float()

    return u / u.norm()

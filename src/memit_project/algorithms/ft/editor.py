from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.utils import nethook

from .hparams import FTHyperParams


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    chunk = []
    for item in arr:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    **_: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **_: Any,
) -> Dict[str, torch.Tensor]:
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"] and request["target_new"]["str"][0] != " ":
            request["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]

    wd = (
        hparams.weight_decay
        if not isinstance(hparams.wd_power_law, tuple)
        else (len(requests) ** hparams.wd_power_law[0]) * np.exp(hparams.wd_power_law[1])
    )
    opt = torch.optim.Adam([v for _, v in weights.items()], lr=hparams.lr, weight_decay=wd)
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    loss_meter = AverageMeter()
    device = next(model.parameters()).device
    for _ in range(hparams.num_steps):
        loss_meter.reset()
        for txt, tgt in zip(chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            unk_id = tok.unk_token_id if tok.unk_token_id is not None else -1
            loss_mask = target_ids != unk_id

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            probs = F.log_softmax(
                model(**inputs).logits[torch.arange(bs, device=device), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(1) / loss_mask.sum(1)
            loss = loss.mean()
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if isinstance(hparams.norm_constraint, float):
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(v, min=weights_copy[k] - eps, max=weights_copy[k] + eps)

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    return deltas

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


def build_edit_batches(
    tok: AutoTokenizer,
    texts: List[str],
    targets: List[str],
    batch_size: int,
    device: torch.device,
):
    batches = []
    for txt_batch, tgt_batch in zip(chunks(texts, batch_size), chunks(targets, batch_size)):
        prompt_examples = tok(
            txt_batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        prompt_last_inds = prompt_examples["attention_mask"].sum(dim=1) - 1

        full_input_ids = []
        full_attention_masks = []
        full_labels = []
        for txt, tgt in zip(txt_batch, tgt_batch):
            target_ids = tok(tgt, add_special_tokens=False)["input_ids"]
            full_ids = tok(txt + tgt, add_special_tokens=False)["input_ids"]
            prompt_len = len(full_ids) - len(target_ids)
            labels = [-100] * prompt_len + target_ids
            full_input_ids.append(full_ids)
            full_attention_masks.append([1] * len(full_ids))
            full_labels.append(labels)

        padded = tok.pad(
            {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_masks,
            },
            return_tensors="pt",
        )
        padded = {k: v.to(device) for k, v in padded.items()}

        labels = torch.full_like(padded["input_ids"], -100)
        for i, label_seq in enumerate(full_labels):
            labels[i, : len(label_seq)] = torch.tensor(label_seq, device=device)

        batches.append(
            {
                "prompt_inputs": prompt_examples,
                "prompt_last_inds": prompt_last_inds,
                "full_inputs": padded,
                "labels": labels,
            }
        )

    return batches


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
    edit_batches = build_edit_batches(tok, texts, targets, hparams.batch_size, device)
    with torch.no_grad():
        for batch in edit_batches:
            prompt_logits = model(**batch["prompt_inputs"]).logits
            batch["ref_log_probs"] = F.log_softmax(
                prompt_logits[
                    torch.arange(prompt_logits.size(0), device=device),
                    batch["prompt_last_inds"],
                ],
                dim=-1,
            ).detach()

    for _ in range(hparams.num_steps):
        loss_meter.reset()
        for batch in edit_batches:
            opt.zero_grad()
            outputs = model(**batch["full_inputs"]).logits
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            nll_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            prompt_logits = model(**batch["prompt_inputs"]).logits
            current_log_probs = F.log_softmax(
                prompt_logits[
                    torch.arange(prompt_logits.size(0), device=device),
                    batch["prompt_last_inds"],
                ],
                dim=-1,
            )
            kl_loss = hparams.kl_factor * F.kl_div(
                current_log_probs,
                batch["ref_log_probs"],
                log_target=True,
                reduction="batchmean",
            )

            loss = nll_loss + kl_loss
            loss_meter.update(loss.item(), n=batch["full_inputs"]["input_ids"].shape[0])

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

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.utils import nethook

from .hparams import MENDHyperParams


class MendRewriteExecutor:
    """
    Minimal MEND-compatible executor.
    If a learned editor checkpoint is provided later, this class can be extended
    to consume it. Without a checkpoint it falls back to a single gradient-based
    local update so the baseline remains runnable on new models.
    """

    def _prepare_requests(self, requests: List[Dict]) -> List[Dict]:
        reqs = deepcopy(requests)
        for request in reqs:
            if request["target_new"]["str"] and request["target_new"]["str"][0] != " ":
                request["target_new"]["str"] = " " + request["target_new"]["str"]
        return reqs

    def _build_batch(self, tok: AutoTokenizer, requests: List[Dict], device: str):
        prompts = [request["prompt"].format(request["subject"]) for request in requests]
        targets = [request["target_new"]["str"] for request in requests]
        full_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, targets)]
        batch = tok(full_texts, padding=True, return_tensors="pt").to(device)
        labels = batch["input_ids"].clone()
        labels[:] = -100
        for i, target in enumerate(targets):
            target_ids = tok(target, add_special_tokens=False)["input_ids"][:]
            seq_len = int(batch["attention_mask"][i].sum().item())
            labels[i, seq_len - len(target_ids) : seq_len] = torch.tensor(
                target_ids, device=device
            )
        return batch, labels

    def _resolve_param_names(self, hparams: MENDHyperParams) -> List[str]:
        if hparams.layers:
            return [
                f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                for layer in hparams.layers
            ]
        raise ValueError(
            "MENDHyperParams must define 'layers' for the runnable fallback editor."
        )

    def _load_checkpoint(self, hparams: MENDHyperParams):
        if not hparams.checkpoint_path:
            return None
        ckpt = Path(hparams.checkpoint_path).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"MEND checkpoint not found: {ckpt}")
        return torch.load(ckpt, map_location="cpu")

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MENDHyperParams,
        copy: bool = False,
        return_orig_weights: bool = False,
        **_: Any,
    ) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
        if copy:
            model = deepcopy(model)

        requests = self._prepare_requests(requests)
        param_names = self._resolve_param_names(hparams)
        params = [nethook.get_parameter(model, name) for name in param_names]
        weights_copy = {
            name: param.detach().clone()
            for name, param in zip(param_names, params)
            if return_orig_weights
        }

        checkpoint = self._load_checkpoint(hparams)
        if checkpoint is None:
            print(
                "No MEND checkpoint provided. Falling back to a single local gradient update."
            )

        nethook.set_requires_grad(False, model)
        for param in params:
            param.requires_grad_(True)

        device = str(next(model.parameters()).device)
        batch, labels = self._build_batch(tok, requests, device)
        outputs = model(**batch)
        loss = F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        with torch.no_grad():
            for name, param in zip(param_names, params):
                grad = param.grad
                if grad is None:
                    continue
                delta = -hparams.lr_scale * grad
                if checkpoint is not None and name in checkpoint:
                    delta = checkpoint[name].to(param.device, dtype=param.dtype) * delta
                param.add_(delta.to(param.dtype))
                param.grad = None
                param.requires_grad_(False)

        return model, weights_copy

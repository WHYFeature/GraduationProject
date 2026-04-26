from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.utils import nethook

from .hparams import MENDHyperParams
from .network import MENDCheckpointMetadata, MENDGradientScaleNetwork


def render_prompt(prompt_text: str, subject: str) -> str:
    if "{}" in prompt_text:
        return prompt_text.format(subject)
    return prompt_text


def ensure_leading_space(text: str) -> str:
    if text and not text.startswith(" "):
        return " " + text
    return text


def build_feature_vector(param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    grad = grad.detach().float()
    param = param.detach().float()
    grad_norm = grad.norm().clamp_min(1e-12)
    param_norm = param.norm().clamp_min(1e-12)
    grad_abs = grad.abs()
    cosine = (grad.flatten() @ param.flatten()) / (grad_norm * param_norm)
    features = torch.tensor(
        [
            torch.log10(grad_norm).item(),
            torch.log10(param_norm).item(),
            grad_abs.mean().item(),
            grad_abs.std().item(),
            cosine.item(),
        ],
        device=grad.device,
        dtype=grad.dtype,
    )
    return features


class MendRewriteExecutor:
    def _prepare_requests(self, requests: List[Dict]) -> List[Dict]:
        reqs = deepcopy(requests)
        for request in reqs:
            request["target_new"]["str"] = ensure_leading_space(
                request["target_new"]["str"]
            )
        return reqs

    def _build_rewrite_batch(
        self, tok: AutoTokenizer, requests: List[Dict], device: str
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        prompts = [render_prompt(r["prompt"], r["subject"]) for r in requests]
        targets = [r["target_new"]["str"] for r in requests]
        full_texts = [f"{prompt}{target}" for prompt, target in zip(prompts, targets)]
        batch = tok(full_texts, padding=True, return_tensors="pt").to(device)
        labels = batch["input_ids"].clone()
        labels[:] = -100

        for i, target in enumerate(targets):
            target_ids = tok(target, add_special_tokens=False)["input_ids"]
            seq_len = int(batch["attention_mask"][i].sum().item())
            labels[i, seq_len - len(target_ids) : seq_len] = torch.tensor(
                target_ids, device=device
            )
        return batch, labels

    def _build_target_loss(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompt_texts: Iterable[str],
        target_text: str,
        device: str,
    ) -> torch.Tensor:
        prompt_texts = list(prompt_texts)
        if not prompt_texts:
            return torch.tensor(0.0, device=device)

        target_text = ensure_leading_space(target_text)
        full_texts = [f"{prompt}{target_text}" for prompt in prompt_texts]
        batch = tok(full_texts, padding=True, return_tensors="pt").to(device)
        labels = batch["input_ids"].clone()
        labels[:] = -100
        target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        for i in range(len(prompt_texts)):
            seq_len = int(batch["attention_mask"][i].sum().item())
            labels[i, seq_len - len(target_ids) : seq_len] = torch.tensor(
                target_ids, device=device
            )
        outputs = model(**batch)
        return F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def _build_locality_kl(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompt_texts: Iterable[str],
        subject: str,
        device: str,
    ) -> torch.Tensor:
        prompt_texts = [render_prompt(p, subject) for p in prompt_texts]
        if not prompt_texts:
            return torch.tensor(0.0, device=device)

        batch = tok(prompt_texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(**batch).logits
            base_last = base_logits[
                torch.arange(base_logits.size(0), device=device),
                batch["attention_mask"].sum(1) - 1,
            ]
            base_log_probs = torch.log_softmax(base_last, dim=-1)

        edited_logits = model(**batch).logits
        edited_last = edited_logits[
            torch.arange(edited_logits.size(0), device=device),
            batch["attention_mask"].sum(1) - 1,
        ]
        edited_log_probs = torch.log_softmax(edited_last, dim=-1)
        return F.kl_div(base_log_probs, edited_log_probs, log_target=True, reduction="batchmean")

    def _resolve_param_names(self, hparams: MENDHyperParams) -> List[str]:
        if not hparams.layers:
            raise ValueError("MENDHyperParams.layers must be defined.")
        return [
            f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            for layer in hparams.layers
        ]

    def _load_editor_network(
        self, hparams: MENDHyperParams, device: str
    ) -> Tuple[MENDGradientScaleNetwork, List[float]]:
        if not hparams.checkpoint_path:
            return None, hparams.scale_candidates or [0.5, 1.0, 2.0]

        ckpt_path = Path(hparams.checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MEND checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        meta = MENDCheckpointMetadata(
            scale_candidates=checkpoint["scale_candidates"],
            feature_dim=checkpoint["feature_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        network = MENDGradientScaleNetwork(
            feature_dim=meta.feature_dim,
            hidden_dim=meta.hidden_dim,
            n_candidates=len(meta.scale_candidates),
        ).to(device)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        return network, meta.scale_candidates

    def _compute_rewrite_grads(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        param_names: List[str],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        params = [nethook.get_parameter(model, name) for name in param_names]
        nethook.set_requires_grad(False, model)
        for param in params:
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad = None

        device = str(next(model.parameters()).device)
        batch, labels = self._build_rewrite_batch(tok, requests, device)
        outputs = model(**batch)
        loss = F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        grads = [param.grad.detach().clone() for param in params]

        for param in params:
            param.grad = None
            param.requires_grad_(False)

        return grads, loss.detach()

    def _predict_layer_scales(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        network: MENDGradientScaleNetwork,
        scale_candidates: List[float],
    ) -> List[float]:
        if network is None:
            return [1.0 for _ in grads]

        features = torch.stack(
            [build_feature_vector(param, grad) for param, grad in zip(params, grads)],
            dim=0,
        )
        scales = network.predict_scales(features, scale_candidates)
        return [float(scale.item()) for scale in scales]

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

        grads, _ = self._compute_rewrite_grads(model, tok, requests, param_names)
        device = str(next(model.parameters()).device)
        network, scale_candidates = self._load_editor_network(hparams, device)
        if network is None:
            print(
                "No learned MEND checkpoint provided. Using neutral per-layer scale=1.0 fallback."
            )

        scales = self._predict_layer_scales(params, grads, network, scale_candidates)

        with torch.no_grad():
            for param, grad, scale in zip(params, grads, scales):
                delta = -hparams.lr_scale * scale * grad.to(param.dtype)
                param.add_(delta)

        return model, weights_copy

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.algorithms.mend import MENDHyperParams, MendRewriteExecutor
from memit_project.algorithms.mend.network import MENDGradientScaleNetwork
from memit_project.algorithms.mend.editor import (
    build_feature_vector,
    render_prompt,
)
from memit_project.datasets import (
    CounterFactDataset,
    CustomRewriteDataset,
    MultiCounterFactDataset,
)
from memit_project.utils.model_config import load_model_config
from memit_project.utils.paths import DATA_DIR, HPARAMS_DIR
from memit_project.utils import nethook


DS_DICT = {
    "cf": CounterFactDataset,
    "mcf": MultiCounterFactDataset,
    "custom": CustomRewriteDataset,
}


def load_requests(ds_name: str, tok, size: int, custom_data_path: str | None):
    ds_class = DS_DICT[ds_name]
    kwargs = dict(tok=tok, size=size)
    if ds_name == "custom":
        kwargs["custom_data_path"] = custom_data_path
    return ds_class(DATA_DIR, **kwargs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--hparams_fname", default="Qwen3-4B.json")
    parser.add_argument("--dataset", choices=["cf", "mcf", "custom"], default="custom")
    parser.add_argument("--custom_data_path", default=None)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--update_hparams_json", default=None)
    args = parser.parse_args()

    hparams_path = HPARAMS_DIR / "mend" / args.hparams_fname
    hparams = MENDHyperParams.from_json(hparams_path)
    model_cfg = load_model_config(args.model_name, args.model_config)
    print(f"Using model config {model_cfg['_config_path']}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token

    ds = load_requests(args.dataset, tok, args.train_size, args.custom_data_path)
    executor = MendRewriteExecutor()
    param_names = executor._resolve_param_names(hparams)
    params = [nethook.get_parameter(model, name) for name in param_names]

    scale_candidates = hparams.scale_candidates or [0.25, 0.5, 1.0, 2.0, 4.0]
    network = MENDGradientScaleNetwork(
        feature_dim=5,
        hidden_dim=hparams.hidden_dim,
        n_candidates=len(scale_candidates),
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    device = str(next(model.parameters()).device)
    history = []

    for epoch in range(args.epochs):
        epoch_losses = []
        for idx, record in enumerate(ds):
            request = {
                "case_id": record["case_id"],
                **record,
                **record["requested_rewrite"],
            }
            request["target_new"]["str"] = executor._prepare_requests([request])[0]["target_new"]["str"]

            grads, _ = executor._compute_rewrite_grads(model, tok, [request], param_names)
            features = torch.stack(
                [build_feature_vector(param, grad) for param, grad in zip(params, grads)],
                dim=0,
            )

            neighborhood_prompts = request.get("neighborhood_prompts", [])[:3]
            base_neighborhood = None
            if neighborhood_prompts:
                rendered = [render_prompt(p, request["subject"]) for p in neighborhood_prompts]
                batch = tok(rendered, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**batch).logits
                    last_logits = logits[
                        torch.arange(logits.size(0), device=device),
                        batch["attention_mask"].sum(1) - 1,
                    ]
                    base_neighborhood = torch.log_softmax(last_logits, dim=-1)

            labels = []
            for layer_idx, (param, grad) in enumerate(zip(params, grads)):
                orig = param.detach().clone()
                candidate_scores = []
                for scale in scale_candidates:
                    with torch.no_grad():
                        param[...] = orig + (-hparams.lr_scale * scale * grad.to(param.dtype))

                    rewrite_loss = executor._build_target_loss(
                        model,
                        tok,
                        [render_prompt(request["prompt"], request["subject"])],
                        request["target_new"]["str"],
                        device,
                    )
                    paraphrase_loss = executor._build_target_loss(
                        model,
                        tok,
                        request.get("paraphrase_prompts", [])[:2],
                        request["target_new"]["str"],
                        device,
                    )
                    neighborhood_kl = torch.tensor(0.0, device=device)
                    if base_neighborhood is not None and neighborhood_prompts:
                        rendered = [render_prompt(p, request["subject"]) for p in neighborhood_prompts]
                        batch = tok(rendered, padding=True, return_tensors="pt").to(device)
                        logits = model(**batch).logits
                        last_logits = logits[
                            torch.arange(logits.size(0), device=device),
                            batch["attention_mask"].sum(1) - 1,
                        ]
                        edited_log_probs = torch.log_softmax(last_logits, dim=-1)
                        neighborhood_kl = F.kl_div(
                            base_neighborhood,
                            edited_log_probs,
                            log_target=True,
                            reduction="batchmean",
                        )
                    score = (
                        rewrite_loss
                        + hparams.paraphrase_weight * paraphrase_loss
                        + hparams.neighborhood_kl_weight * neighborhood_kl
                    )
                    candidate_scores.append(float(score.item()))
                    with torch.no_grad():
                        param[...] = orig

                best_idx = min(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
                labels.append(best_idx)

            labels = torch.tensor(labels, device=device, dtype=torch.long)
            logits = network(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if (idx + 1) % 20 == 0:
                print(
                    f"epoch={epoch} step={idx + 1}/{len(ds)} "
                    f"loss={mean(epoch_losses[-20:]):.4f}"
                )

        history.append({"epoch": epoch, "mean_loss": mean(epoch_losses)})
        print(f"epoch={epoch} mean_loss={history[-1]['mean_loss']:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "scale_candidates": scale_candidates,
            "feature_dim": 5,
            "hidden_dim": hparams.hidden_dim,
            "history": history,
        },
        output_path,
    )
    print(f"Saved MEND editor checkpoint to {output_path}")

    if args.update_hparams_json:
        target_json = Path(args.update_hparams_json)
        data = json.loads(target_json.read_text(encoding="utf-8"))
        data["checkpoint_path"] = str(output_path).replace("\\", "/")
        if "scale_candidates" not in data:
            data["scale_candidates"] = scale_candidates
        if "hidden_dim" not in data:
            data["hidden_dim"] = hparams.hidden_dim
        target_json.write_text(json.dumps(data, indent=4), encoding="utf-8")
        print(f"Updated hparams JSON with checkpoint path: {target_json}")


if __name__ == "__main__":
    main()

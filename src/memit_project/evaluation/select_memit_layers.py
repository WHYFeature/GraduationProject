import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.algorithms.memit import MEMITHyperParams
from memit_project.datasets import CounterFactDataset, KnownsDataset
from memit_project.utils import nethook
from memit_project.utils.paths import DATA_DIR, HPARAMS_DIR, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Causal analysis for selecting the MEMIT layer range. "
            "Implements the corruption-and-restoration tracing idea used in "
            "ROME (2202.05262) and summarizes layer ranges for MEMIT (2210.07229)."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hparams_fname", type=str, required=True)
    parser.add_argument(
        "--dataset",
        choices=["knowns", "counterfact"],
        default="knowns",
        help="Fact source used to measure causal restoration scores.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=100,
        help="Use a small subset first; causal tracing is expensive.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of corrupted replicas per fact.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=3.0,
        help="Multiplier over embedding std for subject corruption noise.",
    )
    parser.add_argument(
        "--fact_token",
        choices=["subject_last", "subject_first"],
        default="subject_last",
        help="Which subject token position to restore.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="4,5,6,7,8",
        help="Comma-separated candidate widths for the recommended contiguous layer span.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to results/layer_selection/<model>.",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip writing a PNG summary plot.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r'[\\\\/:*?"<>|]+', "_", name)


def get_hparams_path(hparams_fname: str) -> Path:
    hparams_path = Path(hparams_fname)
    if hparams_path.exists():
        return hparams_path
    candidate = HPARAMS_DIR / "memit" / hparams_fname
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find hparams file: {hparams_fname}")


def infer_embed_layer_name(model) -> str:
    for name in ["transformer.wte", "gpt_neox.embed_in", "model.embed_tokens"]:
        try:
            nethook.get_module(model, name)
            return name
        except LookupError:
            continue
    raise LookupError("Unable to infer token embedding module for this model.")


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    patterns = [
        r"^transformer\.h\.(\d+)$",
        r"^gpt_neox\.layers\.(\d+)$",
        r"^model\.layers\.(\d+)$",
    ]
    max_idx = -1
    for name, _ in model.named_modules():
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    if max_idx >= 0:
        return max_idx + 1
    raise ValueError("Unable to infer layer count from model.")


def make_inputs(tokenizer, prompts: Sequence[str], device: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def decode_tokens(tokenizer, token_array) -> List[str]:
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring: str) -> Tuple[int, int]:
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, tok in enumerate(toks):
        loc += len(tok)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        raise ValueError(f"Failed to locate subject span: {substring}")
    return tok_start, tok_end


def get_embedding_std(model, tokenizer, subjects: Iterable[str], device: str) -> float:
    embed_layer = infer_embed_layer_name(model)
    values = []
    for subject in subjects:
        inp = make_inputs(tokenizer, [subject], device)
        with nethook.Trace(model, embed_layer) as tr:
            model(**inp)
        values.append(tr.output[0].detach())
    return torch.cat(values, dim=0).std().item()


def predict_answer(model, tokenizer, prompt: str, device: str) -> Tuple[int, float, str]:
    inp = make_inputs(tokenizer, [prompt], device)
    with torch.no_grad():
        logits = model(**inp).logits
    probs = torch.softmax(logits[:, -1, :], dim=1)
    prob, pred = torch.max(probs, dim=1)
    token_id = pred.item()
    return token_id, prob.item(), tokenizer.decode([token_id])


def get_restore_index(subject_range: Tuple[int, int], fact_token: str) -> int:
    if fact_token == "subject_last":
        return subject_range[1] - 1
    if fact_token == "subject_first":
        return subject_range[0]
    raise ValueError(f"Unsupported fact token strategy: {fact_token}")


def trace_with_patch(
    model,
    inp: Dict[str, torch.Tensor],
    states_to_patch: List[Tuple[int, str]],
    answer_token: int,
    subject_range: Tuple[int, int],
    embed_layer: str,
    noise: float,
    replace: bool = False,
):
    rs = np.random.RandomState(1)
    patch_spec = defaultdict(list)
    for token_idx, layer_name in states_to_patch:
        patch_spec[layer_name].append(token_idx)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        if layer == embed_layer:
            begin, end = subject_range
            noise_data = noise * torch.from_numpy(
                rs.randn(x.shape[0] - 1, end - begin, x.shape[2])
            ).to(x.device, dtype=x.dtype)
            if replace:
                x[1:, begin:end] = noise_data
            else:
                x[1:, begin:end] += noise_data
            return x
        if layer not in patch_spec:
            return x
        hidden = untuple(x)
        for token_idx in patch_spec[layer]:
            hidden[1:, token_idx] = hidden[0, token_idx]
        return x

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layer] + list(patch_spec.keys()),
        edit_output=patch_rep,
    ):
        outputs = model(**inp)
    probs = torch.softmax(outputs.logits[1:, -1, :], dim=1).mean(dim=0)
    return probs[answer_token].item()


def load_facts(dataset_name: str, size_limit: Optional[int]):
    if dataset_name == "knowns":
        ds = KnownsDataset(DATA_DIR)
        records = [ds[i] for i in range(min(len(ds), size_limit or len(ds)))]
        return [
            {
                "prompt": record["prompt"],
                "subject": record["subject"],
                "expected": record.get("attribute"),
                "case_id": record.get("known_id", i),
            }
            for i, record in enumerate(records)
        ]

    ds = CounterFactDataset(DATA_DIR, size=size_limit)
    return [
        {
            "prompt": record["requested_rewrite"]["prompt"].format(
                record["requested_rewrite"]["subject"]
            ),
            "subject": record["requested_rewrite"]["subject"],
            "expected": record["requested_rewrite"]["target_true"]["str"],
            "case_id": record["case_id"],
        }
        for record in ds
    ]


def contiguous_span_scores(scores: np.ndarray, width: int) -> Tuple[int, float]:
    if width > len(scores):
        raise ValueError("Width exceeds number of layers.")
    best_start, best_score = 0, float("-inf")
    for start in range(len(scores) - width + 1):
        score = float(scores[start : start + width].mean())
        if score > best_score:
            best_start, best_score = start, score
    return best_start, best_score


def moving_average(scores: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1:
        return scores.copy()
    padded = np.pad(scores, (width // 2, width - 1 - width // 2), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")


def analyze_layers(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = MEMITHyperParams.from_json(get_hparams_path(args.hparams_fname))

    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().to(device)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    nethook.set_requires_grad(False, model)

    facts = load_facts(args.dataset, args.dataset_size_limit)
    if not facts:
        raise ValueError("No facts available for analysis.")

    noise = args.noise_level * get_embedding_std(
        model, tok, [fact["subject"] for fact in facts[: min(64, len(facts))]], device
    )
    num_layers = get_num_layers(model)
    embed_layer = infer_embed_layer_name(model)
    scores_by_layer = [[] for _ in range(num_layers)]
    kept_facts = []

    for fact in facts:
        prompt = fact["prompt"]
        subject = fact["subject"]
        clean_token, high_score, answer = predict_answer(model, tok, prompt, device)

        expected = fact.get("expected")
        if expected is not None and expected.strip() and answer.strip() != expected.strip():
            continue

        inp = make_inputs(tok, [prompt] * (args.samples + 1), device)
        subject_range = find_token_range(tok, inp["input_ids"][0], subject)
        restore_idx = get_restore_index(subject_range, args.fact_token)

        low_score = trace_with_patch(
            model,
            inp,
            [],
            clean_token,
            subject_range,
            embed_layer,
            noise=noise,
        )

        denom = max(high_score - low_score, 1e-8)
        for layer in range(num_layers):
            restored = trace_with_patch(
                model,
                inp,
                [(restore_idx, hparams.rewrite_module_tmp.format(layer))],
                clean_token,
                subject_range,
                embed_layer,
                noise=noise,
            )
            recovery = (restored - low_score) / denom
            scores_by_layer[layer].append(recovery)
        kept_facts.append(
            {
                "case_id": fact["case_id"],
                "prompt": prompt,
                "subject": subject,
                "answer": answer,
                "high_score": high_score,
                "low_score": low_score,
            }
        )

    if not kept_facts:
        raise ValueError(
            "No facts were retained. The model may not predict the expected facts in this dataset."
        )

    avg_scores = np.array(
        [float(np.mean(layer_scores)) if layer_scores else 0.0 for layer_scores in scores_by_layer]
    )
    std_scores = np.array(
        [float(np.std(layer_scores)) if layer_scores else 0.0 for layer_scores in scores_by_layer]
    )
    smoothed_scores = moving_average(avg_scores, width=3)

    recommendations = []
    for width in [int(x) for x in args.window_sizes.split(",") if x.strip()]:
        start, score = contiguous_span_scores(smoothed_scores, width)
        recommendations.append(
            {
                "width": width,
                "start_layer": start,
                "end_layer": start + width - 1,
                "mean_score": score,
            }
        )

    model_tag = sanitize_name(Path(args.model_name).name if Path(args.model_name).exists() else args.model_name)
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "layer_selection" / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": args.model_name,
        "hparams": str(get_hparams_path(args.hparams_fname)),
        "dataset": args.dataset,
        "dataset_size_limit": args.dataset_size_limit,
        "samples": args.samples,
        "noise": noise,
        "fact_token": args.fact_token,
        "rewrite_module_tmp": hparams.rewrite_module_tmp,
        "num_layers": num_layers,
        "n_facts_used": len(kept_facts),
        "layer_scores": avg_scores.tolist(),
        "layer_score_std": std_scores.tolist(),
        "layer_scores_smoothed": smoothed_scores.tolist(),
        "recommendations": recommendations,
        "facts_used": kept_facts[:20],
    }

    summary_path = output_dir / "memit_layer_selection.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.skip_plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
        x = np.arange(num_layers)
        ax.plot(x, avg_scores, label="Average recovery", linewidth=2)
        ax.plot(x, smoothed_scores, label="Smoothed recovery", linewidth=2, linestyle="--")
        for rec in recommendations:
            ax.axvspan(rec["start_layer"], rec["end_layer"], alpha=0.12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized recovery")
        ax.set_title("Causal recovery by MEMIT rewrite layer")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "memit_layer_selection.png")
        plt.close(fig)

    print(f"Used {len(kept_facts)} facts")
    print(f"Saved summary to {summary_path}")
    print("Recommended spans:")
    for rec in recommendations:
        print(
            f"  width={rec['width']}: "
            f"[{rec['start_layer']}, {rec['end_layer']}] "
            f"mean_score={rec['mean_score']:.4f}"
        )


if __name__ == "__main__":
    analyze_layers(parse_args())

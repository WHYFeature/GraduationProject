from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memit_project.evaluation.runner import main as run_experiment  # noqa: E402
from memit_project.utils.model_config import sanitize_model_name  # noqa: E402


BASE_HPARAMS = (
    PROJECT_ROOT
    / "src"
    / "memit_project"
    / "algorithms"
    / "memit"
    / "Qwen3-4B.json"
)
MEMIT_HPARAMS_DIR = BASE_HPARAMS.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused MEMIT hyperparameter sweep for Qwen3-4B and summarize "
            "rewrite/paraphrase/neighborhood performance."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset", dest="ds_name", choices=["cf", "mcf", "zsre"], default="cf")
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--num_edits", type=int, default=1)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Results subdirectory. Defaults to memit_tuning/<model>.",
    )
    parser.add_argument(
        "--preset",
        choices=["focused", "layer_only", "grid"],
        default="focused",
        help=(
            "focused: small recommended sweep; layer_only: only layer windows; "
            "grid: Cartesian product from command-line lists."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help='Layer windows such as "6,7,8,9,10,11" "7,8,9,10,11". Used by grid preset.',
    )
    parser.add_argument("--mom2_update_weights", nargs="*", type=int, default=None)
    parser.add_argument("--v_loss_layers", nargs="*", type=int, default=None)
    parser.add_argument("--kl_factors", nargs="*", type=float, default=None)
    parser.add_argument("--clamp_norm_factors", nargs="*", type=float, default=None)
    return parser.parse_args()


def flatten_numeric(values):
    flat = []
    for item in values:
        if isinstance(item, list):
            flat.extend(flatten_numeric(item))
        elif isinstance(item, (int, float, bool)):
            flat.append(float(item))
    return flat


def summarize_run(run_dir: Path) -> Dict:
    case_files = sorted(run_dir.glob("*_edits-case_*.json"))
    if not case_files:
        raise FileNotFoundError(f"No case result files found in {run_dir}")

    aggregate = {
        "rewrite_acc": [],
        "paraphrase_acc": [],
        "neighborhood_acc": [],
        "time": [],
    }
    for case_file in case_files:
        data = json.loads(case_file.read_text(encoding="utf-8"))
        post = data.get("post", {})
        aggregate["time"].append(float(data.get("time", 0.0)))
        for metric_name, result_key in [
            ("rewrite_acc", "rewrite_prompts_correct"),
            ("paraphrase_acc", "paraphrase_prompts_correct"),
            ("neighborhood_acc", "neighborhood_prompts_correct"),
        ]:
            if result_key in post:
                aggregate[metric_name].extend(flatten_numeric(post[result_key]))

    summary = {
        "run_dir": str(run_dir),
        "n_cases": len(case_files),
        "mean_time": mean(aggregate["time"]) if aggregate["time"] else None,
    }
    for key in ["rewrite_acc", "paraphrase_acc", "neighborhood_acc"]:
        if aggregate[key]:
            summary[key] = mean(aggregate[key])

    score_terms = [
        summary.get("rewrite_acc"),
        summary.get("paraphrase_acc"),
        summary.get("neighborhood_acc"),
    ]
    score_terms = [value for value in score_terms if value is not None]
    if score_terms:
        summary["overall_score"] = mean(score_terms)
    return summary


def latest_run_dir(base_dir: Path) -> Path:
    run_dirs = sorted(
        path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return run_dirs[-1]


def parse_layers(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def focused_configs(base: Dict) -> List[Dict]:
    configs = []

    def add(name: str, **updates):
        cfg = dict(base)
        cfg.update(updates)
        cfg["_tune_name"] = name
        configs.append(cfg)

    add("baseline")
    add("layers_7_11", layers=[7, 8, 9, 10, 11])
    add("layers_8_11", layers=[8, 9, 10, 11])
    add("mom2_10000", mom2_update_weight=10000)
    add("mom2_5000", mom2_update_weight=5000)
    add("loss_32", v_loss_layer=32)
    add("loss_28", v_loss_layer=28)
    add("kl_003", kl_factor=0.03)
    add("clamp_10", clamp_norm_factor=1.0)
    return configs


def layer_only_configs(base: Dict) -> List[Dict]:
    windows = [
        [6, 7, 8, 9, 10, 11],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11],
        [6, 7, 8, 9],
    ]
    return [
        {**base, "layers": layers, "_tune_name": "layers_" + "_".join(map(str, layers))}
        for layers in windows
    ]


def grid_configs(args, base: Dict) -> List[Dict]:
    layers = args.layers or ["6,7,8,9,10,11"]
    mom2_weights = args.mom2_update_weights or [base["mom2_update_weight"]]
    loss_layers = args.v_loss_layers or [base["v_loss_layer"]]
    kl_factors = args.kl_factors or [base["kl_factor"]]
    clamp_factors = args.clamp_norm_factors or [base["clamp_norm_factor"]]

    configs = []
    for layer_str, mom2, loss_layer, kl, clamp in itertools.product(
        layers, mom2_weights, loss_layers, kl_factors, clamp_factors
    ):
        layer_list = parse_layers(layer_str)
        cfg = dict(base)
        cfg.update(
            {
                "layers": layer_list,
                "mom2_update_weight": mom2,
                "v_loss_layer": loss_layer,
                "kl_factor": kl,
                "clamp_norm_factor": clamp,
            }
        )
        cfg["_tune_name"] = (
            f"layers_{'_'.join(map(str, layer_list))}"
            f"_mom2_{mom2}_loss_{loss_layer}_kl_{kl}_clamp_{clamp}"
        ).replace(".", "p")
        configs.append(cfg)
    return configs


def make_configs(args, base: Dict) -> List[Dict]:
    if args.preset == "focused":
        return focused_configs(base)
    if args.preset == "layer_only":
        return layer_only_configs(base)
    return grid_configs(args, base)


def write_summary(output_dir: Path, summaries: List[Dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        summaries,
        key=lambda item: item["metrics"].get("overall_score", float("-inf")),
        reverse=True,
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries, "ranking": ranked}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = ["# MEMIT Tuning", ""]
    lines.append("| Rank | Name | Layers | Rewrite | Paraphrase | Neighborhood | Overall | Mean Time |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for i, item in enumerate(ranked, start=1):
        hp = item["hparams"]
        mt = item["metrics"]
        layers = ",".join(map(str, hp["layers"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    item["name"],
                    layers,
                    fmt(mt.get("rewrite_acc")),
                    fmt(mt.get("paraphrase_acc")),
                    fmt(mt.get("neighborhood_acc")),
                    fmt(mt.get("overall_score")),
                    fmt(mt.get("mean_time")),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    base = json.loads(BASE_HPARAMS.read_text(encoding="utf-8"))
    configs = make_configs(args, base)

    model_tag = sanitize_model_name(Path(args.model_name).name)
    output_name = args.output_name or f"memit_tuning/{model_tag}"
    output_dir = PROJECT_ROOT / "results" / output_name
    hparam_dir = output_dir / "hparams"
    hparam_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx, cfg in enumerate(configs):
        tune_name = cfg.pop("_tune_name")
        temp_hparams_name = f"_tune_memit_{idx:03d}_{tune_name}.json"
        temp_hparams_path = MEMIT_HPARAMS_DIR / temp_hparams_name
        saved_hparams = hparam_dir / temp_hparams_name
        temp_hparams_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        saved_hparams.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[MEMIT TUNE] {idx + 1}/{len(configs)} {tune_name}")
        try:
            run_experiment(
                alg_name="MEMIT",
                model_name=args.model_name,
                hparams_fname=temp_hparams_name,
                ds_name=args.ds_name,
                dataset_size_limit=args.dataset_size,
                continue_from_run=None,
                skip_generation_tests=args.skip_generation_tests,
                generation_test_interval=-1,
                conserve_memory=args.conserve_memory,
                dir_name=f"{output_name}/{tune_name}",
                num_edits=args.num_edits,
                use_cache=args.use_cache,
                model_config=args.model_config,
            )
        finally:
            temp_hparams_path.unlink(missing_ok=True)

        run_dir = latest_run_dir(output_dir / tune_name)
        summaries.append(
            {
                "name": tune_name,
                "hparams": cfg,
                "metrics": summarize_run(run_dir),
            }
        )
        write_summary(output_dir, summaries)

    print(f"Saved MEMIT tuning summary to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memit_project.evaluation.runner import main as run_experiment  # noqa: E402
from memit_project.utils.model_config import sanitize_model_name  # noqa: E402


METHODS = ["FT", "MEND", "ROME", "MEMIT"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run smoke tests and formal comparison for FT, MEND, ROME, and MEMIT "
            "on a Qwen-compatible local model, then summarize the results."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset", dest="ds_name", choices=["cf", "mcf", "zsre"], default="cf")
    parser.add_argument("--formal_dataset_size", type=int, default=100)
    parser.add_argument("--smoke_dataset_size", type=int, default=1)
    parser.add_argument("--num_edits", type=int, default=1)
    parser.add_argument("--generation_test_interval", type=int, default=1)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument(
        "--comparison_name",
        type=str,
        default=None,
        help="Results subdirectory name. Defaults to qwen_comparison/<model>.",
    )
    parser.add_argument(
        "--cleanup_stats_at_end",
        action="store_true",
        help="Also remove cached layer statistics under data/stats after all runs finish.",
    )
    return parser.parse_args()


def hparams_for(method: str) -> str:
    return "Qwen3-4B.json"


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def cleanup_temporary_files(project_root: Path):
    for pattern in ("__pycache__",):
        for path in project_root.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    for pattern in ("*.pyc", "*.pyo", "*.tmp", "*.temp"):
        for path in project_root.rglob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)

    for rel in [".pytest_cache", ".mypy_cache", ".ruff_cache"]:
        remove_path(project_root / rel)

    for path in sorted(project_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def latest_run_dir(base_dir: Path) -> Path:
    run_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return run_dirs[-1]


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
        "reference_score": [],
        "ngram_entropy": [],
        "essence_score": [],
        "time": [],
    }

    for case_file in case_files:
        data = json.loads(case_file.read_text(encoding="utf-8"))
        post = data.get("post", {})
        aggregate["time"].append(float(data.get("time", 0.0)))

        if "rewrite_prompts_correct" in post:
            aggregate["rewrite_acc"].extend(flatten_numeric(post["rewrite_prompts_correct"]))
        if "paraphrase_prompts_correct" in post:
            aggregate["paraphrase_acc"].extend(flatten_numeric(post["paraphrase_prompts_correct"]))
        if "neighborhood_prompts_correct" in post:
            aggregate["neighborhood_acc"].extend(flatten_numeric(post["neighborhood_prompts_correct"]))
        if "reference_score" in post:
            aggregate["reference_score"].append(float(post["reference_score"]))
        if "ngram_entropy" in post:
            aggregate["ngram_entropy"].append(float(post["ngram_entropy"]))
        if "essence_score" in post:
            aggregate["essence_score"].append(float(post["essence_score"]))

    summary = {
        "run_dir": str(run_dir),
        "n_cases": len(case_files),
        "mean_time": mean(aggregate["time"]) if aggregate["time"] else None,
    }
    for key, values in aggregate.items():
        if key == "time":
            continue
        if values:
            summary[key] = mean(values)

    score_terms = [
        summary.get("rewrite_acc"),
        summary.get("paraphrase_acc"),
        summary.get("neighborhood_acc"),
    ]
    score_terms = [x for x in score_terms if x is not None]
    if score_terms:
        summary["overall_score"] = mean(score_terms)

    return summary


def write_summary(output_dir: Path, method_summaries: Dict[str, Dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        method_summaries.items(),
        key=lambda item: item[1].get("overall_score", float("-inf")),
        reverse=True,
    )

    summary_json = {
        "methods": method_summaries,
        "ranking": [
            {
                "method": name,
                "overall_score": metrics.get("overall_score"),
                "rewrite_acc": metrics.get("rewrite_acc"),
                "paraphrase_acc": metrics.get("paraphrase_acc"),
                "neighborhood_acc": metrics.get("neighborhood_acc"),
            }
            for name, metrics in ranked
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# Baseline Comparison", ""]
    if ranked:
        best_name, best_metrics = ranked[0]
        lines.append(
            f"Best overall method: {best_name} "
            f"(overall_score={best_metrics.get('overall_score')})"
        )
        lines.append("")
    lines.append("| Method | Rewrite | Paraphrase | Neighborhood | Overall | Mean Time |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(metrics.get("rewrite_acc")),
                    _fmt(metrics.get("paraphrase_acc")),
                    _fmt(metrics.get("neighborhood_acc")),
                    _fmt(metrics.get("overall_score")),
                    _fmt(metrics.get("mean_time")),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main():
    args = parse_args()
    model_tag = sanitize_model_name(Path(args.model_name).name)
    comparison_name = (
        args.comparison_name
        if args.comparison_name is not None
        else f"qwen_comparison/{model_tag}"
    )

    smoke_root = PROJECT_ROOT / "results" / "_tmp_smoke" / model_tag
    comparison_root = PROJECT_ROOT / "results" / comparison_name
    method_summaries = {}

    print("=== Smoke test stage ===")
    for method in METHODS:
        smoke_dir_name = f"_tmp_smoke/{model_tag}/{method}"
        print(f"[SMOKE] {method}")
        run_experiment(
            alg_name=method,
            model_name=args.model_name,
            hparams_fname=hparams_for(method),
            ds_name=args.ds_name,
            dataset_size_limit=args.smoke_dataset_size,
            continue_from_run=None,
            skip_generation_tests=True,
            generation_test_interval=-1,
            conserve_memory=args.conserve_memory,
            dir_name=smoke_dir_name,
            num_edits=args.num_edits,
            use_cache=False,
            model_config=args.model_config,
        )
        remove_path(smoke_root / method)
        cleanup_temporary_files(PROJECT_ROOT)

    print("=== Formal comparison stage ===")
    for method in METHODS:
        formal_dir_name = f"{comparison_name}/{method}"
        print(f"[FORMAL] {method}")
        run_experiment(
            alg_name=method,
            model_name=args.model_name,
            hparams_fname=hparams_for(method),
            ds_name=args.ds_name,
            dataset_size_limit=args.formal_dataset_size,
            continue_from_run=None,
            skip_generation_tests=args.skip_generation_tests,
            generation_test_interval=args.generation_test_interval,
            conserve_memory=args.conserve_memory,
            dir_name=formal_dir_name,
            num_edits=args.num_edits,
            use_cache=args.use_cache,
            model_config=args.model_config,
        )
        run_dir = latest_run_dir(comparison_root / method)
        method_summaries[method] = summarize_run(run_dir)
        cleanup_temporary_files(PROJECT_ROOT)

    if args.cleanup_stats_at_end:
        remove_path(PROJECT_ROOT / "data" / "stats")

    write_summary(comparison_root, method_summaries)
    print(f"Saved comparison summary to {comparison_root}")


if __name__ == "__main__":
    main()

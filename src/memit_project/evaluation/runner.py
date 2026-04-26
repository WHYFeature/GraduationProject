import json
import re
import shutil
from itertools import islice
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.algorithms.memit import MEMITHyperParams, apply_memit_to_model
from memit_project.datasets import (
    AttributeSnippets,
    CounterFactDataset,
    CustomRewriteDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from memit_project.evaluation.counterfact_metrics import (
    compute_rewrite_quality_counterfact,
)
from memit_project.evaluation.zsre_metrics import compute_rewrite_quality_zsre
from memit_project.utils.model_config import (
    apply_model_config_to_hparams,
    load_model_config,
    sanitize_model_name,
)
from memit_project.utils import nethook
from memit_project.utils.paths import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR

try:
    from memit_project.algorithms.ft import FTHyperParams, apply_ft_to_model
except ImportError:
    FTHyperParams = None
    apply_ft_to_model = None

try:
    from memit_project.algorithms.mend import MENDHyperParams, MendRewriteExecutor
except ImportError:
    MENDHyperParams = None
    MendRewriteExecutor = None

try:
    from memit_project.algorithms.rome import ROMEHyperParams, apply_rome_to_model
except ImportError:
    ROMEHyperParams = None
    apply_rome_to_model = None

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model)
}

if ROMEHyperParams is not None and apply_rome_to_model is not None:
    ALG_DICT["ROME"] = (ROMEHyperParams, apply_rome_to_model)
if FTHyperParams is not None and apply_ft_to_model is not None:
    ALG_DICT["FT"] = (FTHyperParams, apply_ft_to_model)
if MENDHyperParams is not None and MendRewriteExecutor is not None:
    ALG_DICT["MEND"] = (MENDHyperParams, MendRewriteExecutor().apply_to_model)
DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "custom": (CustomRewriteDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def resolve_hparams_path(alg_name: str, hparams_fname: str, run_dir: Path, continue_from_run):
    if continue_from_run is not None:
        return run_dir / "params.json"

    candidate_dirs = [
        HPARAMS_DIR / alg_name,
        HPARAMS_DIR / alg_name.lower(),
        HPARAMS_DIR / alg_name.upper(),
    ]
    for candidate_dir in candidate_dirs:
        candidate = candidate_dir / hparams_fname
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find hparams '{hparams_fname}' for algorithm '{alg_name}' under {HPARAMS_DIR}."
    )


def restore_model_weights(model, weights_copy):
    with torch.no_grad():
        for name, value in weights_copy.items():
            param = nethook.get_parameter(model, name)
            param[...] = value.to(param.device)


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    model_config: str = None,
    custom_data_path: str = None,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = resolve_hparams_path(alg_name, hparams_fname, run_dir, continue_from_run)
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    resolved_model_name = (
        str(Path(model_name).expanduser().resolve())
        if isinstance(model_name, str) and Path(model_name).expanduser().exists()
        else model_name
    )
    model_cfg = load_model_config(
        resolved_model_name if isinstance(resolved_model_name, str) else None,
        model_config,
    )
    apply_model_config_to_hparams(hparams, model_cfg)
    print(f"Executing {alg_name} with parameters {hparams}")
    print(f"Using model config {model_cfg['_config_path']}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model_path = Path(model_name).expanduser()
        resolved_model_name = (
            str(model_path.resolve()) if model_path.exists() else model_name
        )
        model = AutoModelForCausalLM.from_pretrained(resolved_model_name).cuda()
        tok = AutoTokenizer.from_pretrained(resolved_model_name)
        tok.pad_token = tok.eos_token
        model_name = resolved_model_name
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds_kwargs = dict(tok=tok, size=dataset_size_limit)
    if ds_name == "custom":
        ds_kwargs["custom_data_path"] = custom_data_path
    ds = ds_class(DATA_DIR, **ds_kwargs)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{sanitize_model_name(model_name)}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        weights_copy = {}
        eval_start = time()
        try:
            start = time()
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [
                    {
                        "case_id": record["case_id"],
                        **record,
                        **record["requested_rewrite"],
                    }
                    for record in record_chunks
                ],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
            exec_time = time() - start
            print("Execution took", exec_time)

            gen_test_vars = [snips, vec]
            for record in record_chunks:
                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    ),
                }

                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
        finally:
            if weights_copy:
                restore_model_weights(model, weights_copy)

        print("Evaluation took", time() - eval_start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=sorted(ALG_DICT.keys()),
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-xl",
        help="Model to edit. Can be a HuggingFace model id or a local model directory.",
        required=True,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Optional path to a model config file under configs/models.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        default="mcf",
        choices=["mcf", "cf", "zsre", "custom"],
        help="Dataset to perform evaluations on. Use custom with --custom_data_path for user-provided rewrite facts.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to a JSON/JSONL custom rewrite file when --ds_name custom is used.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_config=args.model_config,
        custom_data_path=args.custom_data_path,
    )

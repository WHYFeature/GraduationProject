from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit_project.algorithms.rome.repr_tools import (
    get_reprs_at_idxs,
    get_reprs_at_word_tokens,
    get_words_idxs_in_templates,
)
from memit_project.utils import nethook
from memit_project.utils.model_config import get_hidden_size

from .hparams import MEMITHyperParams


def render_prompt(prompt_text: str, subject: str) -> str:
    if "{}" in prompt_text:
        return prompt_text.format(subject)
    return prompt_text


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def find_subject_token_span(
    prompt_text: str,
    subject: str,
    tok: AutoTokenizer,
) -> Tuple[int, int]:
    """
    Returns the [start, end] token span of the subject in a fully rendered prompt.
    Falls back across a few tokenizer variants to handle spacing differences.
    """

    rendered = render_prompt(prompt_text, subject)
    prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    subject_variants = dedupe_keep_order(
        [
            subject,
            " " + subject,
            subject.strip(),
            " " + subject.strip(),
        ]
    )

    for variant in subject_variants:
        subject_ids = tok(variant, add_special_tokens=False)["input_ids"]
        if not subject_ids:
            continue
        for start in range(len(prompt_ids) - len(subject_ids) + 1):
            if prompt_ids[start : start + len(subject_ids)] == subject_ids:
                return start, start + len(subject_ids) - 1

    raise ValueError(
        f"Unable to find subject token span for subject='{subject}' in prompt='{rendered}'"
    )


def get_lm_head_components(
    model: AutoModelForCausalLM, hparams: MEMITHyperParams
) -> Tuple[torch.Tensor, torch.nn.Module, torch.Tensor]:
    try:
        lm_module = nethook.get_module(model, hparams.lm_head_module)
    except LookupError:
        lm_module = None

    if lm_module is None:
        lm_module = model.get_output_embeddings()

    if lm_module is None or not hasattr(lm_module, "weight"):
        raise LookupError(
            f"Unable to locate output embedding/lm head module for {hparams.lm_head_module}"
        )

    lm_w = lm_module.weight.T
    lm_b = getattr(lm_module, "bias", None)
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    ln_f = nethook.get_module(model, hparams.ln_f_module)
    return lm_w, ln_f, lm_b


def get_hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported layer output type: {type(output)}")


def replace_hidden_tensor(output, new_hidden):
    if isinstance(output, torch.Tensor):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    if isinstance(output, list):
        result = list(output)
        result[0] = new_hidden
        return result
    raise TypeError(f"Unsupported layer output type: {type(output)}")


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f, lm_b = get_lm_head_components(model, hparams)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL prompts.
    # In addition to the canonical request prompt, optimize over paraphrases
    # when available to directly encourage broader generalization.
    target_prefix = tok.decode(target_ids[:-1])
    canonical_rewrite_prompts = [
        context.format(request["prompt"]) + target_prefix
        for context_types in context_templates
        for context in context_types
    ]
    paraphrase_prompts = [
        prompt + target_prefix for prompt in request.get("paraphrase_prompts", [])
    ]
    rewriting_prompts = dedupe_keep_order(
        canonical_rewrite_prompts + paraphrase_prompts
    )

    # Preserve behavior across several subject-conditioned prompts instead of
    # a single generic KL prompt; this tends to help locality without changing
    # the underlying MEMIT objective.
    neighborhood_prompts = request.get("neighborhood_prompts", [])
    base_kl_prompts = dedupe_keep_order(
        [
            request["prompt"],
            *request.get("paraphrase_prompts", [])[:2],
            "{} is a",
        ]
    )
    locality_prompts = dedupe_keep_order(neighborhood_prompts[:3])
    all_prompts = rewriting_prompts + base_kl_prompts + locality_prompts

    input_tok = tok(
        [render_prompt(prompt, request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up.
    # Neighborhood prompts typically do not contain the edited subject, so we
    # anchor their locality constraint at the prompt end (next-token position)
    # instead of trying to recover the current subject span.
    lookup_idxs = []
    rewrite_and_base_kl_count = len(rewriting_prompts) + len(base_kl_prompts)
    for i, prompt in enumerate(all_prompts):
        if i < rewrite_and_base_kl_count:
            lookup_idxs.append(
                find_fact_lookup_idx(
                    prompt,
                    request["subject"],
                    tok,
                    hparams.fact_token,
                    verbose=(i == 0),
                )
            )
        else:
            lookup_idxs.append(int(input_tok["attention_mask"][i].sum().item()) - 1)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros(
        (get_hidden_size(model),), requires_grad=True, device="cuda"
    )
    target_init, kl_distr_init, locality_kl_distr_init = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            hidden = get_hidden_tensor(cur_out)
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = hidden[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                hidden[i, idx, :] += delta

            cur_out = replace_hidden_tensor(cur_out, hidden)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            base_kl_positions = list(
                range(
                    len(rewriting_prompts),
                    len(rewriting_prompts) + len(base_kl_prompts),
                )
            )
            kl_logits = torch.stack(
                [logits[pos, lookup_idxs[pos], :] for pos in base_kl_positions],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

            locality_kl_log_probs = None
            if locality_prompts:
                locality_positions = list(
                    range(
                        len(rewriting_prompts) + len(base_kl_prompts),
                        len(rewriting_prompts)
                        + len(base_kl_prompts)
                        + len(locality_prompts),
                    )
                )
                locality_logits = torch.stack(
                    [
                        logits[pos, lookup_idxs[pos], :]
                        for pos in locality_positions
                    ],
                    dim=0,
                )
                locality_kl_log_probs = torch.nn.functional.log_softmax(
                    locality_logits, dim=1
                )
                if locality_kl_distr_init is None:
                    locality_kl_distr_init = locality_kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = get_hidden_tensor(
            tr[hparams.layer_module_tmp.format(loss_layer)].output
        )[: len(rewriting_prompts)]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        locality_kl_loss = torch.tensor(0.0, device="cuda")
        if locality_prompts and hparams.neighborhood_kl_factor > 0:
            locality_kl_loss = hparams.neighborhood_kl_factor * torch.nn.functional.kl_div(
                locality_kl_distr_init,
                locality_kl_log_probs,
                log_target=True,
                reduction="batchmean",
            )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + locality_kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(locality_kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        if "{}" in prompt:
            ret = get_words_idxs_in_templates(
                tok=tok,
                context_templates=[prompt],
                words=[subject],
                subtoken=fact_token_strategy[len("subject_") :],
            )[0][0]
        else:
            start, end = find_subject_token_span(prompt, subject, tok)
            subtoken = fact_token_strategy[len("subject_") :]
            if subtoken == "last":
                ret = end
            elif subtoken == "first":
                ret = start
            else:
                raise ValueError(f"Unsupported subject subtoken strategy: {subtoken}")
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = render_prompt(prompt, subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret

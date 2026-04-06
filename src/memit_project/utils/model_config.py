import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from memit_project.utils.paths import PROJECT_ROOT


MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
MODEL_CONFIG_FIELDS = (
    "rewrite_module_tmp",
    "layer_module_tmp",
    "mlp_module_tmp",
    "attn_module_tmp",
    "ln_f_module",
    "lm_head_module",
)


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r'[\\\\/:*?"<>|]+', "_", model_name)


def model_name_to_key(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path.name.lower()
    normalized = model_name.replace("\\", "/").rstrip("/").split("/")[-1]
    return normalized.lower()


def resolve_model_config_path(
    model_name: Optional[str] = None, model_config_path: Optional[str] = None
) -> Path:
    if model_config_path is not None:
        path = Path(model_config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        return path

    if model_name is None:
        raise ValueError("Either model_name or model_config_path must be provided.")

    key = model_name_to_key(model_name)
    candidates = [MODEL_CONFIG_DIR / f"{key}.yml", MODEL_CONFIG_DIR / f"{key}.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in list(MODEL_CONFIG_DIR.glob("*.yml")) + list(
        MODEL_CONFIG_DIR.glob("*.yaml")
    ):
        with open(candidate, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        aliases = [str(alias).lower() for alias in data.get("aliases", [])]
        if key in aliases:
            return candidate

    raise FileNotFoundError(
        f"Could not resolve model config for '{model_name}'. "
        f"Expected a file under {MODEL_CONFIG_DIR}."
    )


def load_model_config(
    model_name: Optional[str] = None, model_config_path: Optional[str] = None
) -> Dict[str, Any]:
    path = resolve_model_config_path(model_name, model_config_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(path)
    return data


def apply_model_config_to_hparams(hparams, model_config: Dict[str, Any]):
    for field in MODEL_CONFIG_FIELDS:
        if field in model_config and hasattr(hparams, field):
            setattr(hparams, field, model_config[field])
    return hparams


def get_model_config_value(
    model_config: Optional[Dict[str, Any]], key: str, default: Any = None
) -> Any:
    if model_config is None:
        return default
    return model_config.get(key, default)


def get_hidden_size(model, model_config: Optional[Dict[str, Any]] = None) -> int:
    attrs = get_model_config_value(
        model_config, "hidden_size_attrs", ["n_embd", "hidden_size", "d_model"]
    )
    for attr in attrs:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise AttributeError("Unable to infer hidden size from model config.")


def get_context_length(model, model_config: Optional[Dict[str, Any]] = None) -> int:
    attrs = get_model_config_value(
        model_config,
        "context_length_attrs",
        ["n_positions", "max_position_embeddings"],
    )
    for attr in attrs:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise AttributeError("Unable to infer context length from model config.")


def get_num_layers(model, model_config: Optional[Dict[str, Any]] = None) -> int:
    config_attr = get_model_config_value(
        model_config, "num_layers_attr", "num_hidden_layers"
    )
    if hasattr(model.config, config_attr):
        return getattr(model.config, config_attr)

    layer_template = get_model_config_value(model_config, "layer_module_tmp")
    if layer_template is None:
        raise AttributeError("Unable to infer number of layers from model config.")
    prefix, suffix = layer_template.split("{}", 1)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    max_idx = -1
    for name, _ in model.named_modules():
        match = pattern.match(name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    if max_idx >= 0:
        return max_idx + 1
    raise AttributeError("Unable to infer number of layers from model modules.")


def get_embed_layer_name(model_config: Dict[str, Any]) -> str:
    embed_layer = model_config.get("embed_layer")
    if not embed_layer:
        raise KeyError("Model config must define 'embed_layer'.")
    return embed_layer

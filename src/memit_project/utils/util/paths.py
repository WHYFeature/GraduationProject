from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yml"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "globals.yml"

config_path = CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
with open(config_path, "r", encoding="utf-8") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    PROJECT_ROOT / Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

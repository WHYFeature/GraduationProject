import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


REQUIRED_REWRITE_FIELDS = ["prompt", "subject", "target_new", "target_true"]


class CustomRewriteDataset(Dataset):
    """
    Loads user-provided rewrite requests in the same record shape used by
    CounterFact/MEMIT:

    {
      "case_id": 0,
      "requested_rewrite": {
        "prompt": "The mother tongue of {} is",
        "subject": "Danielle Darrieux",
        "target_new": {"str": "English"},
        "target_true": {"str": "French"},
        "relation_id": "custom"
      },
      "paraphrase_prompts": [],
      "neighborhood_prompts": [],
      "generation_prompts": []
    }

    The file can be JSON list, {"records": [...]}, or JSONL.
    """

    def __init__(
        self,
        data_dir: str,
        custom_data_path: Optional[str] = None,
        size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if custom_data_path is None:
            raise ValueError(
                "CustomRewriteDataset requires --custom_data_path pointing to a JSON or JSONL file."
            )

        path = Path(custom_data_path).expanduser()
        if not path.is_absolute():
            path = Path(data_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Custom rewrite file not found: {path}")

        raw_records = load_custom_records(path)
        self.data = [
            normalize_custom_record(record, idx)
            for idx, record in enumerate(raw_records)
        ]
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded custom rewrite dataset with {len(self)} elements from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_custom_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(
        f"Unsupported custom rewrite file format: {path}. "
        "Expected a JSON list, {'records': [...]}, or JSONL."
    )


def normalize_custom_record(record: dict, idx: int) -> dict:
    rewrite = record.get("requested_rewrite", record)
    if not isinstance(rewrite, dict):
        raise ValueError(f"Record {idx} has invalid requested_rewrite: {record}")

    missing = [field for field in REQUIRED_REWRITE_FIELDS if field not in rewrite]
    if missing:
        raise ValueError(
            f"Record {idx} missing requested_rewrite fields: {missing}. "
            f"Required fields are {REQUIRED_REWRITE_FIELDS}."
        )

    prompt = rewrite["prompt"]
    subject = rewrite["subject"]
    if "{}" not in prompt:
        raise ValueError(
            f"Record {idx} prompt must contain '{{}}' placeholder for the subject: {prompt}"
        )
    if subject not in prompt.format(subject):
        raise ValueError(f"Record {idx} subject does not format into prompt correctly.")

    normalized_rewrite = dict(rewrite)
    normalized_rewrite["target_new"] = normalize_target(rewrite["target_new"], idx, "target_new")
    normalized_rewrite["target_true"] = normalize_target(
        rewrite["target_true"], idx, "target_true"
    )
    normalized_rewrite.setdefault("relation_id", "custom")

    return {
        "case_id": record.get("case_id", idx),
        "requested_rewrite": normalized_rewrite,
        "paraphrase_prompts": record.get("paraphrase_prompts", []),
        "neighborhood_prompts": record.get("neighborhood_prompts", []),
        "generation_prompts": record.get("generation_prompts", []),
        "attribute_prompts": record.get("attribute_prompts", []),
    }


def normalize_target(target, idx: int, field_name: str) -> dict:
    if isinstance(target, str):
        return {"str": target, "id": "custom"}
    if isinstance(target, dict) and "str" in target:
        result = dict(target)
        result.setdefault("id", "custom")
        return result
    raise ValueError(
        f"Record {idx} field {field_name} must be a string or a dict containing 'str'."
    )

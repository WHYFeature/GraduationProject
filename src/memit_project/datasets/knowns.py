import json
from pathlib import Path

from torch.utils.data import Dataset


class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        data_dir = Path(data_dir)
        known_loc = data_dir / "known_1000.json"
        if not known_loc.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {known_loc}. "
                "Place the dataset in data/ before running."
            )

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

import json
import typing
from pathlib import Path

from torch.utils.data import Dataset


class CounterFactDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / (
            "counterfact.json" if not multi else "multi_counterfact.json"
        )
        if not cf_loc.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {cf_loc}. "
                "Place the dataset in data/ before running."
            )

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiCounterFactDataset(CounterFactDataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class Task2SimpleDataset(Dataset):
    def __init__(
        self,
        parquet_file,
    ):
        self.dataframe = pd.read_parquet(parquet_file, engine="pyarrow")
        self.data = self.dataframe[self.dataframe["task"] == "Task2"].reset_index(
            drop=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMo-7B-0724-Instruct-hf"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        inputs = self.tokenizer(row["input"])
        outputs = self.tokenizer(row["output"])

        inputs = torch.tensor(inputs["input_ids"], dtype=torch.int64)
        outputs = torch.tensor(outputs["input_ids"], dtype=torch.int64)

        return inputs, outputs

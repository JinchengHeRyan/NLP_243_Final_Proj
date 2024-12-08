import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class Task2SimpleDataset(Dataset):
    def __init__(self, parquet_file, max_length=512):
        self.dataframe = pd.read_parquet(parquet_file, engine="pyarrow")
        self.data = self.dataframe[self.dataframe["task"] == "Task2"].reset_index(
            drop=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMo-7B-0724-Instruct-hf"
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize with truncation, do not set padding here
        inputs = self.tokenizer(
            row["input"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        outputs = self.tokenizer(
            row["output"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}


if __name__ == "__main__":
    dataset = Task2SimpleDataset(
        "../semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet"
    )
    for i in range(5):
        print(dataset[i])
        break

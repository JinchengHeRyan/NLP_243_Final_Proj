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

        return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}


# Only for testing
if __name__ == "__main__":
    dataset = Task2SimpleDataset(
        "../semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet"
    )

    print("Dataset length: ", len(dataset))

    for data in dataset:
        print(data)
        break

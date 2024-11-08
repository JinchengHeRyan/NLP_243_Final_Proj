import torch
from torch.utils.data import DataLoader
from dataset import Task2SimpleDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class Task2SimpleDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        drop_last=False,
    ):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMo-7B-0724-Instruct-hf"
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )


# Only for testing
if __name__ == "__main__":
    dataset = Task2SimpleDataset(
        "../semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet",
        max_length=128,
    )
    dataloader = Task2SimpleDataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        print("input_ids:", input_ids)
        print("input_ids.shape:", input_ids.shape)
        print("attention_mask:", attention_mask)
        print("attention_mask.shape:", attention_mask.shape)
        print("labels:", labels)
        print("labels.shape:", labels.shape)
        break

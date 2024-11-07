import torch
from torch.utils.data import DataLoader
from dataset import Task2SimpleDataset


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
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        input_ids, output_ids = batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=-100
        )
        output_ids = torch.nn.utils.rnn.pad_sequence(
            output_ids, batch_first=True, padding_value=-100
        )
        return input_ids, output_ids


# Only for testing
if __name__ == "__main__":
    dataset = Task2SimpleDataset(
        "../semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet",
    )
    dataloader = Task2SimpleDataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        input_ids, output_ids = batch
        print("input_ids shape: ", input_ids.shape)
        print("output_ids shape: ", output_ids.shape)
        print("input_ids dtype: ", input_ids.dtype)
        print("output_ids dtype: ", output_ids.dtype)
        break

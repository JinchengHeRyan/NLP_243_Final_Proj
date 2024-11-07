import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = "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"  # Copy token here

## Fetch and load model:
snapshot_download(
    repo_id="llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
    token=hf_token,
    local_dir="semeval25-unlearning-model",
)
model = AutoModelForCausalLM.from_pretrained("semeval25-unlearning-model")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")

## Fetch and load dataset:
snapshot_download(
    repo_id="llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public",
    token=hf_token,
    local_dir="semeval25-unlearning-data",
    repo_type="dataset",
)
retain_train_df = pd.read_parquet(
    "semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet",
    engine="pyarrow",
)  # Retain split: train set
retain_validation_df = pd.read_parquet(
    "semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet",
    engine="pyarrow",
)  # Retain split: validation set
forget_train_df = pd.read_parquet(
    "semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet",
    engine="pyarrow",
)  # Forget split: train set
forget_validation_df = pd.read_parquet(
    "semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet",
    engine="pyarrow",
)  # Forget split: validation set

print(retain_train_df)
print(retain_train_df.head().to_string())

print("retain_train_df shape:", retain_train_df.shape)
print("retain_validation_df shape:", retain_validation_df.shape)
print("forget_train_df shape:", forget_train_df.shape)
print("forget_validation_df shape:", forget_validation_df.shape)

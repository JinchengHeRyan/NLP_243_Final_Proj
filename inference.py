import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import argparse
import os
import json
from utils.utils import get_instance
from utils.average import AverageVal
import dataloader, dataset
from rouge_score import rouge_scorer
from tqdm import tqdm


def compute_rouge(model, data_loader, tokenizer):
    avg_rouge = AverageVal()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    model.eval()

    for i, (input_ids, output_ids) in enumerate(tqdm(data_loader, ncols=80)):
        with torch.no_grad():
            pred_ids = model.generate(input_ids, max_length=128)

            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            output_text = tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )  # ground truth output text

            rouge_l_score = scorer.score(output_text, pred_text)["rougeL"].fmeasure

        avg_rouge.update(rouge_l_score)
        tqdm.write(
            "Rouge-L: {:.4f}\t Rouge-L average: {:.4f}".format(
                rouge_l_score, avg_rouge.avg
            )
        )

    return avg_rouge.avg


def main(config, model_dir):
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(model_dir)

    retain_valid_dataset = get_instance(dataset, config["retain_valid_dataset"])
    retain_valid_dataloader = get_instance(
        dataloader, config["retain_valid_dataloader"], retain_valid_dataset
    )

    forget_valid_dataset = get_instance(dataset, config["forget_valid_dataset"])
    forget_valid_dataloader = get_instance(
        dataloader, config["forget_valid_dataloader"], forget_valid_dataset
    )

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")

    model, retain_valid_dataloader, forget_valid_dataloader = accelerator.prepare(
        model, retain_valid_dataloader, forget_valid_dataloader
    )

    retain_valid_rougeL_avg = compute_rouge(model, retain_valid_dataloader, tokenizer)
    forget_valid_rougeL_avg = compute_rouge(model, forget_valid_dataloader, tokenizer)

    print("Retain valid Rouge-L average: {:.4f}".format(retain_valid_rougeL_avg))
    print("Forget valid Rouge-L average: {:.4f}".format(forget_valid_rougeL_avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLM Unlearning Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of the model to inference.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file path (default: None)",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.config), "No such file: %s" % args.config

    # Read config of the whole system.
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(config, args.model_dir)

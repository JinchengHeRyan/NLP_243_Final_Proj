import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataloader
import dataset
from utils.utils import get_instance
from Trainer import Trainer

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main(config, name, print_freq):
    accelerator = Accelerator(device_placement=True, split_batches=True)

    model = AutoModelForCausalLM.from_pretrained(config["original_model_path"])

    # Get Training Datasets and Dataloaders
    retain_train_dataset = get_instance(dataset, config["retain_train_dataset"])
    retain_train_dataloader = get_instance(
        dataloader, config["retain_train_dataloader"], retain_train_dataset
    )
    forget_train_dataset = get_instance(dataset, config["forget_train_dataset"])
    forget_train_dataloader = get_instance(
        dataloader, config["forget_train_dataloader"], forget_train_dataset
    )

    # Get Validation Datasets and Dataloaders
    retain_valid_dataset = get_instance(dataset, config["retain_valid_dataset"])
    retain_valid_dataloader = get_instance(
        dataloader, config["retain_valid_dataloader"], retain_valid_dataset
    )
    forget_valid_dataset = get_instance(dataset, config["forget_valid_dataset"])
    forget_valid_dataloader = get_instance(
        dataloader, config["forget_valid_dataloader"], forget_valid_dataset
    )

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_pretrain_name"])

    optimizer = get_instance(torch.optim, config["optimizer"], model.parameters())

    (
        model,
        optimizer,
        retain_train_dataloader,
        forget_train_dataloader,
        retain_valid_dataloader,
        forget_valid_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer,
        retain_train_dataloader,
        forget_train_dataloader,
        retain_valid_dataloader,
        forget_valid_dataloader,
    )

    # Set checkpoint directory
    chkpt_dir = os.path.join("chkpt", name)
    os.makedirs(chkpt_dir, exist_ok=True)

    # Save the config in the checkpoint directory
    with open(os.path.join(chkpt_dir, "config.json"), "w") as wfile:
        json.dump(config, wfile, indent=4, sort_keys=False)

    # Set logger
    log_dir = os.path.join("logs", name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime("%Y-%m-%d-%H%M.log", time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    logger.info(model)
    logger.info("-" * 50)

    trainer = Trainer(
        accelerator=accelerator,
        chkpt_dir=chkpt_dir,
        model=model,
        optimizer=optimizer,
        retain_trainloader=retain_train_dataloader,
        forget_trainloader=forget_train_dataloader,
        retain_validloader=retain_valid_dataloader,
        forget_validloader=forget_valid_dataloader,
        logger=logger,
        epochs=config["epochs"],
        print_freq=print_freq,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unlearning Sensitive Content")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        required=True,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="experiment",
        type=str,
        required=False,
        help="name of the saved model",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="The frequency of printing information.",
    )
    args = parser.parse_args()

    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(config, args.name, args.print_freq)

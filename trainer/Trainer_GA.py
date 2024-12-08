from transformers.trainer_utils import is_main_process

from utils.average import AverageVal
import time
from accelerate import Accelerator


class Trainer_GA:
    def __init__(
        self,
        accelerator: Accelerator,
        chkpt_dir,
        model,
        optimizer,
        retain_trainloader,
        forget_trainloader,
        retain_validloader,
        forget_validloader,
        logger,
        epochs,
        print_freq=10,
    ):
        self.accelerator = accelerator
        self.chkpt_dir = chkpt_dir
        self.model = model
        self.optimizer = optimizer
        self.retain_trainloader = retain_trainloader
        self.forget_trainloader = forget_trainloader
        self.retain_validloader = retain_validloader
        self.forget_validloader = forget_validloader
        self.logger = logger
        self.epochs = epochs
        self.print_freq = print_freq

    def _train_mix_ga_da_epoch(self, epoch):
        self.model.train()
        losses = AverageVal()

        retain_losses = AverageVal()
        forget_losses = AverageVal()

        datatime = 0
        batchtime = 0
        tic = time.time()

        for i, (retain_batch, forget_batch) in enumerate(
            zip(self.retain_trainloader, self.forget_trainloader)
        ):
            datatime += time.time() - tic

            retain_input_ids = retain_batch["input_ids"]
            retain_attention_mask = retain_batch["attention_mask"]
            retain_labels = retain_batch["labels"]

            forget_input_ids = forget_batch["input_ids"]
            forget_attention_mask = forget_batch["attention_mask"]
            forget_labels = forget_batch["labels"]

            retain_outputs = self.model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
                labels=retain_labels,
            )
            forget_outputs = self.model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                labels=forget_labels,
            )

            retain_loss = retain_outputs.loss
            forget_loss = forget_outputs.loss

            loss = (
                retain_loss
                - 0.5 * (retain_loss.item() / forget_loss.item()) * forget_loss
            )

            losses.update(loss.item())
            retain_losses.update(retain_loss.item())
            forget_losses.update(forget_loss.item())

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            batchtime += time.time() - tic
            tic = time.time()

            if (i + 1) % self.print_freq == 0:
                self.logger.info(
                    "[Train] Epoch:{} [{:03d}/{:03d}]({:.0f}%)\t"
                    "Time:{:.3f}/{:.3f}\tLoss:({:.4f}){:.4f}\tRetain_loss:({:.4f}){:.4f}\tForget_loss:({:.4f}){:.4f}".format(
                        epoch,
                        i + 1,
                        len(
                            list(zip(self.retain_trainloader, self.forget_trainloader))
                        ),
                        (i + 1)
                        / len(
                            list(zip(self.retain_trainloader, self.forget_trainloader))
                        )
                        * 100,
                        datatime,
                        batchtime,
                        losses.val,
                        losses.avg,
                        retain_losses.val,
                        retain_losses.avg,
                        forget_losses.val,
                        forget_losses.avg,
                    )
                )
                datatime = 0
                batchtime = 0

        self.logger.info(
            "[Train Summary] Epoch:{}\tTotal Loss:{:.4f}\t Retain Loss:{:.4f}\t Forget Loss:{:.4f}".format(
                epoch, losses.avg, retain_losses.avg, forget_losses.avg
            )
        )
        return losses.avg

    def _train_ga_da_epoch(self, dataloader, operation, epoch):
        assert operation in ["ga", "gd"]

        self.model.train()
        losses = AverageVal()

        datatime = 0
        batchtime = 0
        tic = time.time()

        for i, batch in enumerate(dataloader):
            datatime += time.time() - tic

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            losses.update(loss.item())

            if operation == "ga":
                loss = -loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            batchtime += time.time() - tic
            tic = time.time()

            if (i + 1) % self.print_freq == 0:
                self.logger.info(
                    "[Train] Epoch:{} [{:03d}/{:03d}]({:.0f}%)\t"
                    "Time:{:.3f}/{:.3f}\tOperation [{}]\tLoss:({:.4f}){:.4f}".format(
                        epoch,
                        i + 1,
                        len(dataloader),
                        (i + 1) / len(dataloader) * 100,
                        datatime,
                        batchtime,
                        operation,
                        losses.val,
                        losses.avg,
                    )
                )
                datatime = 0
                batchtime = 0

        return losses.avg

    def _train_epoch(self, epoch):
        retain_loss = self._train_ga_da_epoch(self.retain_trainloader, "gd", epoch)
        forget_loss = self._train_ga_da_epoch(self.forget_trainloader, "ga", epoch)
        self.logger.info(
            "[Train Summary] Epoch:{}\t Retain Loss:{:.4f}\t Forget Loss:{:.4f}".format(
                epoch, retain_loss, forget_loss
            )
        )

    def _eval_epoch(self):
        pass

    def train(self):
        for epoch in range(self.epochs):
            self._train_mix_ga_da_epoch(epoch)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            self.chkpt_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )

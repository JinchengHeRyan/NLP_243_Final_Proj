from utils.average import AverageVal
import time


class Trainer:
    def __init__(
        self,
        accelerator,
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
        self.model = model
        self.optimizer = optimizer
        self.retain_trainloader = retain_trainloader
        self.forget_trainloader = forget_trainloader
        self.retain_validloader = retain_validloader
        self.forget_validloader = forget_validloader
        self.logger = logger
        self.epochs = epochs
        self.print_freq = print_freq

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
            self._train_epoch(epoch)

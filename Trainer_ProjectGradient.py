from utils.average import AverageVal
import time
import torch
import os


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

    def project_gradients(self, grad_sensitive, grad_normal):
        """
        Project gradients to minimize influence of sensitive data.
        """
        projected_grads = []
        for g_s, g_n in zip(grad_sensitive, grad_normal):
            if g_s is not None and g_n is not None:
                projection = torch.dot(g_s.flatten(), g_n.flatten()) / (torch.norm(g_n.flatten()) ** 2 + 1e-8)
                g_s_proj = g_s - projection * g_n
                projected_grads.append(g_s_proj)
            else:
                projected_grads.append(g_s)  # Use original gradient if no projection needed
        return projected_grads

    def randomize_forget_set(self, dataloader):
        """
        Randomly modifies the labels (answers) in the forget set.
        """
        for batch in dataloader.dataset:
            if "labels" in batch:
                labels_tensor = torch.tensor(batch["labels"])  # Convert list to tensor
                randomized_labels = torch.randint(
                    high=self.model.config.vocab_size, size=labels_tensor.size()
                )
                batch["labels"] = randomized_labels.tolist()

    def _train_ga_da_epoch(self, dataloader, operation, epoch):
        assert operation in ["ga", "gd", "pgd", "ran"]

        self.model.train()
        losses = AverageVal()

        datatime = 0
        batchtime = 0
        tic = time.time()

        grad_sensitive = None
        grad_normal = None

        for i, batch in enumerate(dataloader):
            datatime += time.time() - tic

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if operation == "ran":
                self.randomize_forget_set(dataloader)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            losses.update(loss.item())

            if operation == "ga":
                loss = -loss
            if operation == "pgd":
                if grad_sensitive is not None and grad_normal is not None:
                    projected_grads = self.project_gradients(grad_sensitive, grad_normal)
                    for param, grad in zip(self.model.parameters(), projected_grads):
                        if grad is not None:
                            param.grad = grad
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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
        ran_loss = self._train_ga_da_epoch(self.forget_trainloader, "ran", epoch)
        retain_loss = self._train_ga_da_epoch(self.retain_trainloader, "gd", epoch)
        # forget_loss = self._train_ga_da_epoch(self.forget_trainloader, "ga", epoch)
        pgd_loss = self._train_ga_da_epoch(self.forget_trainloader, "pgd", epoch)
        self.logger.info(
            "[Train Summary] Epoch:{}\t Ran Loss:{:.4f}\t Retain Loss:{:.4f}\t PGD Loss:{:.4f}".format(
                epoch, ran_loss, retain_loss, pgd_loss)
        )
        weight_ran = 0.25
        weight_retain = 0.4
        weight_forget = 0.1
        weight_pgd = 0.25

        combined_loss = weight_ran * ran_loss + weight_retain * retain_loss + weight_pgd * pgd_loss

        return {"combined_loss": combined_loss, "ran_loss": ran_loss, "retain_loss": retain_loss, "pgd_loss": pgd_loss}

    def _eval_epoch(self):
        pass

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()  # Ensure all processes are synchronized
        self.accelerator.unwrap_model(self.model).save_pretrained(output_dir)
        self.logger.info(f"Model saved to {output_dir}")

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
        self.save_model("NLP_243_Final_Proj/output_model")
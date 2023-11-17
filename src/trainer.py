import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        test_data,
        loss_fn,
        optimizer,
        config,
        generator=torch.Generator().manual_seed(1337),
        
    ):

        if model == None or train_data == None:
            raise Exception("Model and train_data must be specified")

        if test_data == None:
            if "split_test" not in config:
                raise Exception("Must specify either test_data or split_train")
            else:
                # warnings.warn("Test data is not specified, will split train data into train and validation")
                train_len = int(len(train_data) * (1 - config["split_test"]))
                test_len = len(train_data) - train_len

            if config["split_random"]:
                train_data, test_data = random_split(train_data, (train_len, test_len))

            else:
                train_data, test_data = random_split(
                    train_data, (train_len, test_len), generator=generator
                )

        if optimizer == None:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-3, weight_decay=1e-8
            )
        else:
            self.optimizer = optimizer

        if "device" not in config:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config["device"]

        if "name" not in config:
            self.name = datetime.now().strftime("%Y-%m-%d_%H")
        else:
            self.name = config["name"]

        if "test_metrics" not in config:
            self.test_metrics = [loss_fn]
        else:
            self.test_metrics = config["test_metrics"]

        if "test_metric_names" not in config:
            self.test_metric_names = [
                f"Metric {i}" for i in range(len(self.test_metrics))
            ]
        else:
            assert len(config["test_metric_names"]) == len(self.test_metrics)
            self.test_metric_names = config["test_metric_names"]

        if "scheduler" not in config:
            self.scheduler = None
        else:
            self.scheduler = config["scheduler"]

        if "tensorboard_path" in config:
            self.writer = SummaryWriter(log_dir=config["tensorboard_path"])
        else:
            self.writer = None

        self.batch_size = config["batch_size"]
        self.num_epochs = config["epochs"]
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=config["shuffle"], num_workers=16, persistent_workers=True, pin_memory=True
        )
        self.test_dataloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=config["shuffle"], num_workers=16, presistent_workers=True, pin_memory=True
        )
        self.epochs_between_safe = config["epochs_between_safe"]
        self.batches_between_safe = config["batches_between_safe"]

        self.test_scores = pd.DataFrame(columns=(["epoch"] + self.test_metric_names))
        self.train_loss = pd.DataFrame(
            columns=["Epoch", "Batch", "Curr Loss", "Running Loss"]
        )

    def train_loop(self, epoch=0):
        size = len(self.train_dataloader.dataset)
        time_at_start = time.time() * 1000
        
        self.model.train()
        running_loss = np.array([])

        with tqdm(self.train_dataloader, desc=f"Epoch {epoch}") as pbar:
            pbar_dict = {}
            for batch, (input, y) in enumerate(pbar):
                self.optimizer.zero_grad()
                pred = self.model(input.to(self.device))
                y = y.to(self.device)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss = np.append(running_loss, loss.item())
                if (batch * self.batch_size) % (size * 0.1) < self.batch_size:
                    current = batch * len(input[0])
                    pbar_dict["Running Loss"] = running_loss.sum()/(batch + 1)
                    run_time = time.time() * 1000 - time_at_start
                    pbar.set_postfix(pbar_dict)
                    if self.writer is not None:
                        self.writer.add_scalar("Running Loss/train", running_loss.sum()/(batch + 1), epoch * len(self.train_dataloader) + batch)
                if (
                    self.batches_between_safe is not None
                    and batch % self.batches_between_safe == 0
                    and batch > self.batches_between_safe - 1
                ):
                    print("saving model...")
                    torch.save(self.model.state_dict(), f"models/tmp_model_weights.pth")
            return running_loss

    def test_loop(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss = [0 for _ in self.test_metrics]
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.test_dataloader, desc="Test") as pbar:
                pbar_dict = {}
                for batch, (input, y) in enumerate(pbar):
                    pred = self.model(input.to(self.device))
                    y = y.to(self.device)
                    for i, metric in enumerate(self.test_metrics):
                        loss = metric(pred, y)
                        test_loss[i] += loss.item()
                    if (batch * self.batch_size) % (size * 0.2) < self.batch_size:
                        for i, metric in enumerate(self.test_metrics):
                            loss = test_loss[i] / (batch + 1)
                            pbar_dict[self.test_metric_names[i]] = loss
                            if self.writer is not None:
                                self.writer.add_scalar(f"Running Loss/test_{self.test_metric_names[i]}", loss, batch)
                        pbar.set_postfix(pbar_dict)
        if self.scheduler != None:
            self.scheduler.step(test_loss[0])
        avg_test_loss = {}
        for i, metric in enumerate(self.test_metrics):
            loss = test_loss[i] / num_batches
            avg_test_loss[self.test_metric_names[i]] = loss
            print(f"{self.test_metric_names[i]}: {loss:>7f}")
            if self.writer is not None:
                self.writer.add_scalar(f"Loss/test_{self.test_metric_names[i]}", loss, batch)
        return avg_test_loss

    def train_test(self):
        try:
            print(f"Training model with name: {self.name}")
            for t in range(self.num_epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                running_loss = self.train_loop()
                current_test_loss = self.test_loop()
                current_test_loss["epoch"] = t
                print("Current test loss:\n", current_test_loss)

                try:
                    if (
                        current_test_loss["DiceLoss"]
                        < self.test_scores["DiceLoss"].min()
                    ):
                        print("New best model")
                        torch.save(
                            self.model.state_dict(),
                            f"models/best_model_weights_{self.name}.pth",
                        )
                    if ((t + 1) % 5) == 0:
                        print("Saving model")
                        torch.save(
                            self.model.state_dict(),
                            f"models/{self.name}_epoch_{t+1}.pth",
                        )

                except Exception as e:
                    print(e)

                self.test_scores = pd.concat(
                    [
                        self.test_scores,
                        pd.DataFrame(current_test_loss, index=["epoch"]),
                    ]
                )
                print("")
            print("Done!")
            if t % self.epochs_between_safe == 0:
                torch.save(
                    self.model.state_dict(), f"models/model_weights_{self.name}.pth"
                )
            # torch.save(self.model, f'models/entire_model_{self.name}.pth')
        except KeyboardInterrupt:
            print("Abort...")
            safe = input("Safe model [y]es/[n]o: ")
            if safe == "y" or safe == "Y":
                torch.save(
                    self.model.state_dict(), f"models/model_weights_{self.name}.pth"
                )
                # torch.save(self.model, f'models/entire_model_{self.name}.pth')
            else:
                print("Not saving...")

        torch.save(self.model.state_dict(), f"models/model_weights_{self.name}.pth")
        return self.test_scores
        # torch.save(self.model, f'models/entire_model_{self.name}.pth')

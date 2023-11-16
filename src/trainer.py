import torch
import time
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        train_data,
        test_data,
        loss_fn,
        optimizer,
        split_test: float = None,
        device=None,
        batch_size=32,
        epochs=10,
        shuffle=True,
        name=None,
        test_metrics=None,
        test_metric_names=None,
        scheduler=None,
        epochs_between_safe=1,
        batches_between_safe=None,
        split_random=False,
        generator=torch.Generator().manual_seed(1337),
    ):
        if model == None or train_data == None:
            raise Exception("Model and train_data must be specified")

        if test_data == None:
            if split_test == None:
                raise Exception("Must specify either test_data or split_train")
            elif split_test != None:
                # warnings.warn("Test data is not specified, will split train data into train and validation")
                train_len = int(len(train_data) * (1 - split_test))
                test_len = len(train_data) - train_len

            if split_random:
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

        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if name == None:
            self.name = datetime.now().strftime("%Y-%m-%d_%H")
        else:
            self.name = name

        if test_metrics == None:
            self.test_metrics = [loss_fn]
        else:
            self.test_metrics = test_metrics

        if test_metric_names == None:
            self.test_metric_names = [
                f"Metric {i}" for i in range(len(self.test_metrics))
            ]
        else:
            assert len(test_metric_names) == len(self.test_metrics)
            self.test_metric_names = test_metric_names

        if scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", patience=2
            )
        else:
            self.scheduler = scheduler

        self.batch_size = batch_size
        self.num_epochs = epochs
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=16, persistent_workers=True
        )
        self.test_dataloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=16
        )
        self.epochs_between_safe = epochs_between_safe
        self.batches_between_safe = batches_between_safe

        self.test_scores = pd.DataFrame(columns=(["epoch"] + self.test_metric_names))
        self.train_loss = pd.DataFrame(
            columns=["Epoch", "Batch", "Curr Loss", "Running Loss"]
        )

    def train_loop(self):
        size = len(self.train_dataloader.dataset)
        time_at_start = time.time() * 1000
        self.model.train()
        running_loss = np.array([])
        for batch, (input, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            pred = self.model(input.to(self.device))
            y = y.to(self.device)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss = np.append(running_loss, loss.item())
            if (batch * self.batch_size) % (size * 0.1) < self.batch_size:
                current = batch * len(input[0])
                print(
                    f"loss: {(running_loss.sum()/(batch + 1)):>7f}  [{current:>5d}/{size:>5d}]"
                )
                run_time = time.time() * 1000 - time_at_start
                print(
                    f"time running: {run_time}, time per elem: {run_time/(current+1)}"
                )

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
            for batch, (input, y) in enumerate(self.test_dataloader):
                pred = self.model(input.to(self.device))
                y = y.to(self.device)
                for i, metric in enumerate(self.test_metrics):
                    loss = metric(pred, y)
                    test_loss[i] += loss.item()
                if (batch * self.batch_size) % (size * 0.25) < self.batch_size:
                    for i, metric in enumerate(self.test_metrics):
                        loss = test_loss[i] / (batch + 1)
                        print(
                            f"{self.test_metric_names[i]}: {loss:>7f}  [{batch:>5d}/{num_batches:>5d}]"
                        )

        if self.scheduler != None:
            self.scheduler.step(test_loss[0])
        avg_test_loss = {}
        for i, metric in enumerate(self.test_metrics):
            loss = test_loss[i] / num_batches
            avg_test_loss[self.test_metric_names[i]] = loss
            print(f"{self.test_metric_names[i]}: {loss:>7f}")
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
                    print(f"Baka")
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

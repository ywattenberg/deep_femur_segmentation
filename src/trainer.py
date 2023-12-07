import torch
import time
import os
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from monai.inferers.inferer import SlidingWindowInfererAdapt
import matplotlib.pyplot as plt

def collate_fn(batch):
  return [
      torch.stack([x[0] for x in batch]),
      torch.stack([x[1] for x in batch])
    ]

class Trainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        loss_fn,
        optimizer,
        config,
        generator=torch.Generator().manual_seed(1337),
        test_data=None,
        
    ):
        whole_config = config.copy()
        if "trainer" in config:
            config = config["trainer"]

        if model == None or train_data == None:
            raise Exception("Model and train_data must be specified")

        if val_data == None:
            if "split_test" not in config:
                raise Exception("Must specify either val_data or split_train")
            else:
                # warnings.warn("Test data is not specified, will split train data into train and validation")
                train_len = int(len(train_data) * (1 - config["split_test"]))
                test_len = len(train_data) - train_len

            if config["split_random"]:
                train_data, val_data = random_split(train_data, (train_len, test_len))

            else:
                train_data, val_data = random_split(
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
            self.test_metrics = []
            self.test_metric_names = config["test_metrics"]
            for metric in config["test_metrics"]:
                if metric.upper() == "MSELoss" or metric.upper() == "MSE":
                    self.test_metrics.append(torch.nn.MSELoss())

        if "scheduler" not in config:
            self.scheduler = None
        else:
            self.scheduler = config["scheduler"]

        if "tensorboard_path" in config:
            runs = os.listdir(config["tensorboard_path"])
            runs = [int(run.split("_")[-1]) for run in runs if "run_" in run]
            runs = sorted(runs)
            if len(runs) == 0:
                run_num = 0
            else:
                run_num = runs[-1] 
            log_dir = os.path.join(config["tensorboard_path"], f"run_{run_num}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"Writing to {log_dir}")
            self.writer.add_text("Config", str(whole_config))
        else:
            self.writer = None
        
        self.test_dataset = test_data
        self.batch_size = config["batch_size"]
        self.num_epochs = config["epochs"]
        self.model = model.to(self.device)
        self.loss_fn = loss_fn

        self.train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=config["shuffle"], num_workers=config["num_workers"], persistent_workers=True
        )
        self.val_dataloader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=config["shuffle"], num_workers=config["num_workers"],
            persistent_workers=True
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
        
        with tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
            pbar_dict = {}
            for batch, (input, y) in enumerate(pbar):
                self.optimizer.zero_grad()
                pred = self.model(input.to(self.device))
                y = y.to(self.device)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss = np.append(running_loss, loss.item())
                if (batch * self.batch_size) % (size * 0.01) < self.batch_size:
                    current = batch * len(input[0])
                    pbar_dict["Running Loss"] = running_loss.sum()/(batch + 1)
                    run_time = time.time() * 1000 - time_at_start
                    pbar.set_postfix(pbar_dict)
                    if self.writer is not None:
                        self.writer.add_scalar("Avg. Running Loss/train", running_loss.sum()/(batch + 1), epoch * len(self.train_dataloader) + batch)
                        self.writer.add_scalar("Current Loss/train", running_loss[-1], epoch * len(self.train_dataloader) + batch)
                if (
                    self.batches_between_safe is not None
                    and batch % self.batches_between_safe == 0
                    and batch > self.batches_between_safe - 1
                ):
                    print("saving model...")
                    torch.save(self.model.state_dict(), f"models/tmp_model_weights.pth")
            return running_loss

    def test_loop(self, epoch=0):
        size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        test_loss = [0 for _ in self.test_metrics]
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.val_dataloader, desc="Test") as pbar:
                pbar_dict = {}
                for batch, (input, y) in enumerate(pbar):
                    pred = self.model(input.to(self.device))
                    y = y.to(self.device)
                    for i, metric in enumerate(self.test_metrics):
                        loss = metric(pred, y)
                        test_loss[i] += loss.item()
    
                        pbar.set_postfix(pbar_dict)
        if self.scheduler != None:
            self.scheduler.step(test_loss[0])
        avg_test_loss = {}
        for i, metric in enumerate(self.test_metrics):
            loss = test_loss[i] / num_batches
            avg_test_loss[self.test_metric_names[i]] = loss
            print(f"{self.test_metric_names[i]}: {loss:>7f}")
            if self.writer is not None:
                self.writer.add_scalar(f"Loss/test_{self.test_metric_names[i]}", loss, epoch)
        return avg_test_loss

    def train_test(self, epoch=0):
        try:
            print(f"Training model with name: {self.name}")
            for t in range(epoch, self.num_epochs+ epoch):
                print(f"Epoch {t + 1}\n-------------------------------")
                running_loss = self.train_loop(t)
                current_test_loss = self.test_loop(t)
                current_test_loss["epoch"] = t
                print("Current test loss:\n", current_test_loss)
                if ((t + 1) % self.epochs_between_safe) == 0:
                    print("Saving model")
                    torch.save(
                        self.model.state_dict(),
                        f"models/{self.name}_epoch_{t+1}.pth",
                    )
                self.test_scores = pd.concat(
                    [
                        self.test_scores,
                        pd.DataFrame(current_test_loss, index=["epoch"]),
                    ]
                )
                self.qualitative_eval(t)
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

    def qualitative_eval(self, epoch=0):
        if self.writer is None or self.test_dataset is None:
            return
        self.model.eval()
        inferer = SlidingWindowInfererAdapt(roi_size=(64,64,64), sw_batch_size=8, overlap=0.25, mode="gaussian")
        dataset = self.test_dataset
        slices = [250, len(dataset) - 500]
        for i in slices:
            x,y = dataset[i]

            x = x.to("cuda")
            y = y.to("cuda")
            x = x.unsqueeze(0)
            pred = inferer(x, self.model)
            pred = pred.squeeze().to("cpu").detach().numpy()
            x = x.to("cpu").detach().numpy()
            y = y.to("cpu").detach().numpy()
            fig, ax = plt.subplots(2,2)
            half_depth_x = int(x.shape[0]/2)
            half_depth_y = int(y.shape[0]/2)

            ax[0,0].imshow(x.squeeze()[half_depth_x,:,:])
            ax[0,0].title.set_text("Input")
            ax[1,0] .imshow(y.squeeze()[half_depth_y,:,:])
            ax[1,0].title.set_text("Target")
            ax[1,1].imshow(pred[half_depth_y,:,:])
            ax[1,1].title.set_text("Prediction")
            self.writer.add_figure(f"Qualitative eval/slice_{i}", fig, global_step=epoch)

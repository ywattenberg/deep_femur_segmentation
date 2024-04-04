import numpy as np
import torch
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import SlidingWindowInfererAdapt
import matplotlib.pyplot as plt
import lightning as L


class LitBaseSegmentationTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.configure_model()
        self.loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        iou_metric = MeanIoU(include_background=False)
        self.metrics = {"dice": dice_metric, "iou": iou_metric}

    def configure_model(self):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
        return optimizer, scheduler


    def pred_to_mask(self, y_hat, theshold=0.7):
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > theshold).float()
        return y_hat

    def calculate_metrics(self, y_hat, y):
        y_hat = self.pred_to_mask(y_hat)
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric(y_hat, y).aggregate().item()
        return result

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self.forward(x)
            loss = self.loss(y_hat, y)
            metrics = self.calculate_metrics(y_hat, y)
            self.log("val_loss", loss)
            for name, value in metrics.items():
                self.log(f"val_{name}", value)
            if batch_idx == 0:
                data = self.plot_results(x, y, y_hat, slice_idx=32)
                self.logger.experiment.add_image("val_results", data, self.current_epoch)

    def plot_batch(self, x, y, y_hat, slice_idx=0):
        batch_size = x.shape[0]
        fig, ax = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        for i in range(batch_size):
            ax[i, 0].imshow(x[i, 0, slice_idx].cpu().numpy(), cmap="gray")
            ax[i, 0].axis("off")
            ax[i, 1].imshow(y[i, 0, slice_idx].cpu().numpy(), cmap="gray")
            ax[i, 1].axis("off")
            ax[i, 2].imshow(y_hat[i, 0, slice_idx].cpu().numpy(), cmap="gray")
            ax[i, 2].axis("off")
        ax[0, 0].set_title("Input")
        ax[0, 1].set_title("Ground Truth")
        ax[0, 2].set_title("Prediction")
        plt.tight_layout()



    def plot_results(self, x, y, y_hat, slice_idx=0):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(x[0, slice_idx].cpu().numpy(), cmap="gray")
        ax[0].set_title("Input")
        ax[0].axis("off")
        ax[1].imshow(y[0, slice_idx].cpu().numpy(), cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")
        ax[2].imshow(y_hat[0, slice_idx].cpu().numpy(), cmap="gray")
        ax[2].set_title("Prediction")
        ax[2].axis("off")
        plt.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            predictor = SlidingWindowInfererAdapt(roi_size=self.config["input_size"], sw_batch_size=1, overlap=0.25, mode="gaussian", progress=False)
            y_hat = predictor(self.model, x)
            metrics = self.calculate_metrics(y_hat, y)
            for name, value in metrics.items():
                self.log(f"test_{name}", value)
            if batch_idx == 0:
                data = self.plot_results(x, y, y_hat)
                self.logger.experiment.add_image("test_results", data, self.current_epoch)

class LitUNetSegmentationTrainer(LitBaseSegmentationTrainer):

    def __init__(self, config):
        super().__init__(config)

    def configure_model(self):
        model = monai_nets.UNet(
                spatial_dims=self.config["spatial_dims"],
                in_channels=1,
                out_channels=1,
                channels=self.config["features"],
                strides=self.config["strides"],
                dropout=self.config["dropout"],
                norm=self.config["norm"],
                act=self.config["activation"],
            )
        return model
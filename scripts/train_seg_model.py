import torch.autograd.gradcheck
import yaml
import os
import sys
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import torch


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.pylightning import LitUNetSegmentationTrainer
from src.dataset.dataset_segmentation_full import FemurImageSegmentationDataset

def main():
    torch.set_float32_matmul_precision('medium')
    # torch.set_float16_matmul_precision('medium')
    CONFIG = yaml.safe_load(open("config/config.yaml", "r"))
    # model = LitUNetSegmentationTrainer(CONFIG)
    # trainer = Trainer(precision="16-mixed")
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, mode="binsearch")

    # trainer.fit(model)
    # return



    dataset = FemurImageSegmentationDataset(config=CONFIG, split="train")
    val_conf = CONFIG.copy()
    val_conf["context_csv_path"] = "data/validation.csv"
    val_dataset = FemurImageSegmentationDataset(config=val_conf, split="val")

    train_loader = DataLoader(dataset, batch_size=CONFIG["trainer"]["batch_size"], shuffle=True, num_workers=CONFIG["trainer"]["num_workers"], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["trainer"]["batch_size"], shuffle=False, num_workers=CONFIG["trainer"]["num_workers"], pin_memory=True, persistent_workers=True)

    model = LitUNetSegmentationTrainer(CONFIG)
    model = LitUNetSegmentationTrainer.load_from_checkpoint(r"lightning_logs\95fwd5ym\checkpoints\epoch=11-step=1740.ckpt")
    logger = WandbLogger(name="unet")
    logger.watch(model)
    trainer = Trainer(check_val_every_n_epoch=1, enable_checkpointing=True, logger=logger, precision="16-mixed", default_root_dir="models/", max_epochs=1)
    # trainer.validate(model, val_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()





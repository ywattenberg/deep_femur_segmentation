import torch
import numpy as np
import pandas as pd
import tqdm
import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import SimpleITK as sitk
from monai.inferers.inferer import SlidingWindowInfererAdapt, SlidingWindowInferer
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.networks.nets import UNet, BasicUNetPlusPlus
from monai.networks import nets

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.dataset.dataset_segmentation_full import FemurImageSegmentationDataset 
from src.model.basic_UNet import BasicUNet, UpsampleUNet
from src.model.retina_UNet import Retina_UNet
from src.trainer.pylightning import LitUNetSegmentationTrainer


def predict_index(model, dataset, index):
    input = dataset[index][0]
    input = input.unsqueeze(0).to("cuda")
    with torch.no_grad():
        predictor = SlidingWindowInfererAdapt(roi_size=[64,64,64], sw_batch_size=1, overlap=0.25, mode="gaussian", progress=True)
        pred_mask = predictor(input, model)
    # pred_mask = sliding_window_inference(input, roi_size=[64,64,64], sw_batch_size=8, predictor=model, overlap=0.25, mode="gaussian")
    return torch.nn.functional.sigmoid(pred_mask[0]).to('cpu').detach().squeeze().numpy(), input.squeeze().to('cpu').detach().numpy()

def safe_image(image, path, spacing=(1,1,1)):
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, path)

def main(model_path, config, output_path):
    dataset = FemurImageSegmentationDataset(config=config, split="test")
    model = UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    # # model = Retina_UNet(in_channels=1, out_channels_mask=2, out_channels_upsample=1, config=config)
    # model.load_state_dict(torch.load(model_path))
    # model = LitUNetSegmentationTrainer.load_from_checkpoint(model_path)
    model.eval()
    model.to("cuda")
    for i in range(0,len(dataset), 250):
        print(f"Predicting slice {i}")
        mask, img = predict_index(model, dataset, i)

        img = img > -0.3
        mask = mask > 0.5
        mask = img
        safe_image(mask.astype(np.uint8) , os.path.join(output_path, f"pred_mask_{i}.nii.gz"))
        input, target = dataset[i]
        safe_image(input.squeeze(), os.path.join(output_path, f"input_{i}.nii.gz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    main(args.model_path, config, args.output_path)
import torch
import numpy as np
import pandas as pd
import tqdm
import os
import sys
import matplotlib.pyplot as plt
from monai.inferers.inferer import SlidingWindowInfererAdapt, SlidingWindowInferer
from monai.inferers.utils import compute_importance_map, sliding_window_inference


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset.dataset_segmentation import FemurSegmentationDataset
from src.model.basic_UNet import BasicUNet, UpsampleUNet
from monai.networks.nets import UNet

config = {'context_csv_path': r'HRpQCT_aim\numpy\Cropped_regions_val.csv', 'input_size': [64, 64, 64], 'output_size': [128, 128, 128], 'use_accelerator': True, 'augmentation': True, 'use_cortical_and_trabecular': True, 'base_path': r'C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data', 'pcct_path': 'PCCT', 'mask_path': r'HRpQCT_aim\numpy', 'seed': 2001, 'augmentation_params': {'mask_threshold': 0.92, 'pcct_intensity_scale': [0, 3, -1, 1], 'hrpqct_intensity_scale': [-1.5, 1.5, -1, 1], 'rotation_range': [45, 45, 45], 'p_rotation': 0.8, 'min_zoom': 0.9, 'max_zoom': 1.1, 'p_zoom': 0.5, 'p_flip': 0.5, 'noise_std': 0.1, 'p_noise': 0.5, 'smooth_sigma': [0.25, 0.1], 'p_smooth': 0.5, 'p_intensity_scale': 0.5, 'intensity_scale_factors': 0.2, 'p_intensity_shift': 0.5, 'intensity_shift_offsets': 0.1, 'p_contrast': 0.3, 'contrast_gamma': 4}, 'model': {'spatial_dims': 3, 'features': [32, 64, 64, 128, 256], 'strides': [2, 2, 2, 2], 'dropout': 0.3, 'activation': 'ReLU', 'bias': True, 'norm': 'BATCH'}, 'trainer': {'split_test': 0.2, 'device': 'cuda', 'batch_size': 16, 'num_workers': 8, 'epochs': 200, 'shuffle': True, 'name': 'Segmentation_UNet', 'test_metrics': ['Dice'], 'epochs_between_safe': 1, 'batches_between_safe': 200, 'split_random': False, 'tensorboard_path': 'tensorboard'}}

def inference_by_index(slices):
    # config = yaml.safe_load(open("config/config.yaml", "r"))
    # config["context_csv_path"] = "data/validation.csv"
    dataset = FemurSegmentationDataset(config=config, split="test")

    model = UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=2 if config["use_cortical_and_trabecular"] else 1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    model.load_state_dict(torch.load("models/model_weights_Segmentation_UNet.pth"))
    # model
    model.eval()
    model.to("cuda")

    # upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    # upsample.to("cuda")

    # df = pd.DataFrame(columns=["run", "L1_model", "MSE_model", "L1_baseline", "MSE_baseline"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    
    inferer = SlidingWindowInferer(roi_size=[64,64,64], mode="gaussian", sw_batch_size=1, overlap=0.25, sw_device="cuda", device="cuda", progress=True)
    torch.cuda.empty_cache()


    for i in slices:
        print(f"Slice {i}")
        x,y,mask = dataset[i]
        x = x[:64]
        y = y[:64]
        mask = mask[:64]
        
        print(f"Slice {i}")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {mask.shape}")

        x = x.unsqueeze(0)
        output = inferer(x, model)
        output = torch.nn.functional.sigmoid(output).to('cpu').detach().squeeze().numpy()
        half_depth_x = int(x.shape[0]/2)
        half_depth_y = int(y.shape[0]/2)
        # upsample_output = upsample(x[:,half_depth_x-1:half_depth_x+2]).detach().squeeze().numpy()
        print(f"Output shape: {output.shape}")
        x = x.detach().numpy()
        y = y.detach().numpy()
        x = x.squeeze()
        y = y.squeeze()


        # fig, ax = plt.subplots(2,3)
        # print(f"Half depth: {output[half_depth_y,:,:].shape}")
        # ax[0,0].imshow(x.squeeze()[half_depth_x,:,:], cmap="gray")
        # ax[0,0].title.set_text("Input")
        # ax[1,0].imshow(y.squeeze()[half_depth_y,:,:], cmap="gray")
        # ax[1,0].title.set_text("Target")
        # ax[0,1].imshow(upsample_output[1,:,:], cmap="gray")
        # ax[0,1].title.set_text("Baseline Upsample (NN)")
        # ax[1,1].imshow(output[half_depth_y,:,:], cmap="gray")
        # ax[1,1].title.set_text("Prediction")
        # ax[0,2].imshow(output[half_depth_y, :, :], cmap="gray")
        # ax[0,2].imshow(upsample_output[1,:,:], cmap="Blues", alpha=0.5)
        # ax[0,2].title.set_text("Prediction and Baseline")
        # ax[1,2].imshow(output[half_depth_y,:,:] - upsample_output[half_depth_y,:,:], cmap="jet")
        # ax[1,2].title.set_text("Prediction - Baseline")
        
        # # Add colorbar
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(ax[1,2].imshow(output[half_depth_y,:,:] - upsample_output[half_depth_y,:,:], cmap="jet"), cax=cbar_ax)
        # plt.show()

        fig, ax = plt.subplots(3,2)
        ax[0,0].imshow(x[half_depth_x,:,:], cmap="gray")  
        ax[0,0].title.set_text("Input")
        ax[1,0].imshow(mask[0, half_depth_y,:,:], cmap="gray")
        ax[1,0].title.set_text("Target 1 ")
        ax[1,1].imshow(output[0,half_depth_y,:,:], cmap="gray")
        ax[1,1].title.set_text("Prediction 1") 
        ax[2,0].imshow(mask[1, half_depth_y,:,:], cmap="gray")
        ax[2,0].title.set_text("Target 2")
        ax[2,1].imshow(output[1, half_depth_y,:,:], cmap="gray")
        ax[2,1].title.set_text("Prediction 2 ")    

        plt.show()




def main():
    inference_by_index([0, 1])


if __name__ == "__main__":
    main()
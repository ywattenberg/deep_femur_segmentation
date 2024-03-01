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
from src.dataset.dataset import FemurImageDataset
from src.model.basic_UNet import BasicUNet, UpsampleUNet

config = {'context_csv_path': 'data/train.csv', 'input_size': [64, 64, 64], 'output_size': [128, 128, 128], 'use_accelerator': True, 'augmentation': False, 'base_path': './', 'seed': 2001, 'dtype': 'float32', 'augmentation_params': {'rotation_range': [45, 45, 45], 'p_rotation': 0.8, 'min_zoom': 0.9, 'max_zoom': 1.1, 'p_zoom': 0.5, 'p_flip': 0.5, 'noise_std': 0.1, 'p_noise': 0.5, 'smooth_sigma': [0.25, 0.1], 'p_smooth': 0.5, 'p_intensity_scale': 0.5, 'intensity_scale_factors': 0.2, 'p_intensity_shift': 0.5, 'intensity_shift_offsets': 0.1, 'p_contrast': 0.3, 'contrast_gamma': 4}, 'model': {'spatial_dims': 3, 'features': [64, 64, 128, 128, 256, 256], 'upsample_features': 64, 'strides': [2, 2, 2, 2, 2, 2], 'dropout': 0.3, 'activation': 'ReLU', 'bias': True, 'norm': 'Instance'}, 'trainer': {'device': 'cuda', 'batch_size': 16, 'num_workers': 8, 'epochs': 10, 'shuffle': True, 'name': 'NewUNet_No_Augmentation', 'test_metrics': ['L1', 'MSE'], 'epochs_between_safe': 1, 'batches_between_safe': 200, 'split_random': False, 'tensorboard_path': 'tensorboard'}}

def inference_by_index(slices):
    # config = yaml.safe_load(open("config/config.yaml", "r"))
    config["context_csv_path"] = "data/validation.csv"
    dataset = FemurImageDataset(config=config, split="test")

    model = UpsampleUNet(1,1, config)
    model.load_state_dict(torch.load("models/model_weights_NewUNet_No_Augmentation.pth"))
    # model
    model.eval()

    upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    upsample.to("cuda")

    df = pd.DataFrame(columns=["run", "L1_model", "MSE_model", "L1_baseline", "MSE_baseline"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    # for i in range(5):
    #     baseline_l1 = 0
    #     baseline_mse = 0
    #     model_l1 = 0
    #     model_mse = 0

    #     for x,y in tqdm.tqdm(dataloader):

    #         x = x.to("cuda")
    #         y = y.to("cuda")
    #         output = model(x)
    #         baseline = upsample(x)
    #         l1_model = torch.nn.L1Loss()(output, y).item()
    #         mse_model = torch.nn.MSELoss()(output, y).item()
    #         l1_baseline = torch.nn.L1Loss()(baseline, y).item()
    #         mse_baseline = torch.nn.MSELoss()(baseline, y).item()
    #         baseline_l1 += l1_baseline
    #         baseline_mse += mse_baseline
    #         model_l1 += l1_model
    #         model_mse += mse_model
    #     baseline_l1 /= len(dataset)
    #     baseline_mse /= len(dataset)
    #     model_l1 /= len(dataset)
    #     model_mse /= len(dataset)
    #     print({"run": i, "L1_model": model_l1, "MSE_model": model_mse, "L1_baseline": baseline_l1, "MSE_baseline": baseline_mse})
    #     new_row = {"run": i, "L1_model": model_l1, "MSE_model": model_mse, "L1_baseline": baseline_l1, "MSE_baseline": baseline_mse}
    #     df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    # print(df)
    
    inferer = SlidingWindowInferer(roi_size=[64,64,64], mode="gaussian", sw_batch_size=1, overlap=0.25, sw_device="cpu", device="cpu")
    torch.cuda.empty_cache()


    for i in slices:
        x,y = dataset[i]
        print(f"Slice {i}")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")

        # divide image into for patches as it is too large
        x
        x = x.unsqueeze(0)
        output = inferer(x, model)
        output = output.detach().squeeze().numpy()
        half_depth_x = int(x.shape[0]/2)
        half_depth_y = int(y.shape[0]/2)
        upsample_output = upsample(x[:,half_depth_x-1:half_depth_x+2]).detach().squeeze().numpy()
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

        fig, ax = plt.subplots(1,3)
        # ax[0].imshow(output[half_depth_y,:,:] - y[half_depth_y,:,:], cmap="gray")
        ax[0].title.set_text("Prediction - Target")
        # Colorbar for prediction - target
        fig.colorbar(ax[0].imshow(output[half_depth_y,:,:] - y[half_depth_y,:,:], cmap="jet"), ax=ax[0])
        # ax[1].imshow(output[half_depth_y,:,:] - upsample_output[half_depth_y,:,:], cmap="gray")
        ax[1].title.set_text("Baseline - Target")
        fig.colorbar(ax[1].imshow(upsample_output[half_depth_y,:,:]- y[half_depth_y,:,:], cmap="jet"), ax=ax[1])
        ax[2].title.set_text("Prediction - Baseline")
        # Colorbar for prediction - baseline
        fig.colorbar(ax[2].imshow(output[half_depth_y,:,:] - upsample_output[half_depth_y,:,:], cmap="jet"), ax=ax[2])

        plt.show()




def main():
    inference_by_index([150, 500, 800])


if __name__ == "__main__":
    main()
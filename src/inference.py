import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from monai.inferers.inferer import SlidingWindowInfererAdapt


from dataset.dataset import FemurImageDataset
from model.basic_UNet import BasicUNet, UpsampleUNet

config = {'context_csv_path': 'data/train.csv', 'input_size': [64, 64, 64], 'output_size': [128, 128, 128], 'use_accelerator': True, 'augmentation': True, 'base_path': './', 'seed': 2001, 'augmentation_params': {'rotation_range': [45, 45, 45], 'p_rotation': 0.8, 'min_zoom': 0.9, 'max_zoom': 1.1, 'p_zoom': 0.5, 'p_flip': 0.5, 'noise_std': 0.1, 'p_noise': 0.5, 'smooth_sigma': [0.25, 0.1], 'p_smooth': 0.5, 'p_intensity_scale': 0.5, 'intensity_scale_factors': 0.2, 'p_intensity_shift': 0.5, 'intensity_shift_offsets': 0.1, 'p_contrast': 0.3, 'contrast_gamma': 4}, 'model': {'name': 'BasicUNet', 'spatial_dims': 3, 'features': [32, 32, 64, 128, 256, 32], 'strides': [2, 2, 2, 2, 2], 'activation': 'ReLU', 'dropout': 0.3, 'bias': True, 'norm': 'Instance'}, 'trainer': {'device': 'cuda', 'batch_size': 16, 'num_workers': 8, 'epochs': 10, 'shuffle': True, 'name': 'BasicUNet', 'test_metrics': ['MSE'], 'epochs_between_safe': 1, 'batches_between_safe': 200, 'split_random': False, 'tensorboard_path': 'tensorboard'}}



def inference_by_index(slices):
    # config = yaml.safe_load(open("config/config.yaml", "r"))
    config["context_csv_path"] = "data/train.csv"
    dataset = FemurImageDataset(config=config, split="test")

    # valid_p_size = ensure_tuple(valid_patch_size)
    # importance_map_ = compute_importance_map(
    #     valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
    # )
    # if len(importance_map_.shape) == 3 and not process_fn:
    #     importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions


    inferer = SlidingWindowInfererAdapt(roi_size=(64,64,64), sw_batch_size=1, overlap=0.1, mode="gaussian", cache_roi_weight_map=True)

    model = UpsampleUNet(1,1, config)
    model.load_state_dict(torch.load("models/BasicUNet_epoch_8.pth"))
    model.to("cuda")
    model.eval()

    for i in slices:
        x,y = dataset[i]
        print(f"Slice {i}")
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")

        x = x.to("cuda")
        y = y.to("cuda")
        x = x.unsqueeze(0)
        pred = inferer(x, model)
        print(f"Prediction shape: {pred.shape}")
        pred = pred.squeeze().to("cpu").detach().numpy()
        x = x.to("cpu").detach().numpy()
        y = y.to("cpu").detach().numpy()
        fig, ax = plt.subplots(2,2)
        half_depth_x = int(x.shape[0]/2)
        half_depth_y = int(y.shape[0]/2)
        print(f"Half depth: {pred[half_depth_y,:,:].shape}")
        ax[0,0].imshow(x.squeeze()[half_depth_x,:,:])
        ax[0,0].title.set_text("Input")
        ax[1,0].imshow(y.squeeze()[half_depth_y,:,:])
        ax[1,0].title.set_text("Target")
        ax[1,1].imshow(pred[half_depth_y,:,:])
        ax[1,1].title.set_text("Prediction")
        plt.show()

def main():
    inference_by_index([150, 250, 500, 800])


if __name__ == "__main__":
    main()
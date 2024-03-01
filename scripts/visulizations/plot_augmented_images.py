import torch
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    RandSpatialCropd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandRotated,
    RandZoomd,
    RandAffined,
    RandFlipd,
    ToTensord,
    SpatialCropd,
    HistogramNormalized
)


def main():

    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    config["input_size"] = [100, 500, 500]
    config["output size"] = [100, 500, 500]

    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation_range']]

    # Calculate the initial crops size such that after rotation we don't lose any information
    inner_lenght = np.ceil(np.sqrt((config["output size"][0] ** 2 + config["output size"][1] ** 2)))
    inital_crop_size = int(np.ceil(np.sqrt((inner_lenght ** 2 + config["output size"][2] ** 2))))+1

    target_folder = r"test_out"
    image = sitk.ReadImage(r"data\PCCT\8_2210_06492_L\result.mha")
    image = sitk.GetArrayFromImage(image[:,:, 0:1000])

    img_min = np.min(image)
    img_max = np.max(image)
    img_mean = np.mean(image)
    img_std = np.std(image)

    image = (image - img_mean)  / img_std

    perc_99_5 = np.percentile(image, 99.5)
    perc_00_5 = np.percentile(image, 00.5)
    config['statistics'] = {"intensities": {}} 
    config['statistics']['intensities']['percentile_99_5'] = perc_99_5
    config['statistics']['intensities']['percentile_00_5'] = perc_00_5

    image = image.astype(np.float32)
    plt.imshow(image[int(image.shape[0]/2), :,:], cmap="gray")
    plt.savefig(f"{target_folder}/original.png", dpi=300)



    transforms = [
    RandSpatialCropd(keys=['image'], roi_size=[inital_crop_size, inital_crop_size, inital_crop_size], random_size=False),
    ScaleIntensityRanged(keys=['image'], a_min=perc_00_5, a_max=perc_99_5, b_min=-1, b_max=1.0, clip=True),
    HistogramNormalized(keys=['image'], num_bins=256, min=-1, max=1),
    RandRotated(keys=['image'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
    RandZoomd(keys=['image' ], min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
    RandFlipd(keys=['image' ], prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
    RandFlipd(keys=['image' ], prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
    RandFlipd(keys=['image' ], prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
    SpatialCropd(keys=['image' ], roi_center=[int(inital_crop_size/2) for i in range(3)], roi_size=config["output size"]),
    RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
    RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
    RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
    RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
    RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
]
    # Define transforms
    image = np.expand_dims(image, axis=0)
    img_dict = {"image": image}
    for i, transform in enumerate(transforms):
        img_dict = transform(img_dict)
        img = img_dict["image"]
        print(f"Transform: {transform.__class__.__name__}, img shape: {img.shape}")
        # Plot center slice of image after transform
        plt.imshow(img[0, int(img.shape[1]/2), :, :], cmap="gray")
        plt.savefig(f"{target_folder}/{i}_{transform.__class__.__name__}.png", dpi=300)


if __name__ == "__main__":
    main()

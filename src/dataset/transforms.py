import numpy as np
from torchvision.transforms import v2
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
    SpatialCropd
)

def get_image_augmentation(config, split):
    # Rotation angles in radians
    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation_range']]

    # Calculate the initial crops size such that after rotation we don't lose any information
    inner_lenght = np.ceil(np.sqrt((config["output size"][0] ** 2 + config["output size"][1] ** 2)))
    inital_crop_size = int(np.ceil(np.sqrt((inner_lenght ** 2 + config["output size"][2] ** 2))))+1

    if split == "train":
        transforms = Compose([
            RandSpatialCropd(keys=['image', 'labels'], roi_size=[inital_crop_size, inital_crop_size, inital_crop_size], random_size=False),
            ScaleIntensityRanged(keys=['image', 'labels'], a_min=config['statistics']['intensities']['percentile_00_5'], a_max=config['statistics']['intensities']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True),
            RandRotated(keys=['image', 'labels'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
            RandZoomd(keys=['image', 'labels'], min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
            SpatialCropd(keys=['image', 'labels'], roi_center=[int(inital_crop_size/2) for i in range(3)], roi_size=config["output size"]),
            RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
            RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
            RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
            RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
            RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
            ToTensord(keys=['image', 'labels'])
        ])
    elif split == "val":
        transforms = Compose([
            # ScaleIntensityRanged(keys=['image', 'labels'],a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output size"], random_size=False),
            ToTensord(keys=['image', 'labels'])
            ])
        print("VAL")
    elif split == "test":
        pass

    return transforms




    


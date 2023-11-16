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
    ToTensord
)

def get_image_augmentation(config, split):
    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation']]
    # translate_range = [(i * config['augmentation_params']['translate']) for i in config['statistics']['shape']['median']]

    if split == "train":
        transforms = Compose([
            ScaleIntensityRanged(keys=['image', 'labels'], a_min=config['statistics']['intensities']['percentile_00_5'], a_max=config['statistics']['intensities']['percentile_99_5'], b_min=0.0, b_max=1.0, clip=True),
            RandRotated(keys=['image', 'labels'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
            RandZoomd(keys=['image', 'labels'], min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
            RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output size"], random_size=False),
            RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
            RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma=config["augmentation_params"]["smooth_sigma"]),
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




    


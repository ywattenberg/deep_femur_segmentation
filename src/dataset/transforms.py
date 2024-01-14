import numpy as np
from functools import partial
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
    SpatialCropd,
    Crop,
    Randomizable,
    CenterSpatialCropd,
    HistogramNormalized
)
import logging
from src.dataset.utils import get_inital_crop_size

class CustomCropRandomd(Randomizable, Crop):
    def __init__(self, keys, roi_size_image, roi_size_label):
        super().__init__()
        self.roi_size_image = roi_size_image
        self.roi_size_label = roi_size_label
        self.scale_factor = [b/a for a,b in zip(self.roi_size_image, self.roi_size_label)]
        self.keys = keys
    
    def __call__(self, data, lazy=False):
        img_shape = data[self.keys[0]].shape[-3:]
        # Get Random Center Point for Crop
        half_roi_size = [int(i/2) for i in self.roi_size_image]
        center = [0,0,0]
        if  half_roi_size[0] >= img_shape[0] - half_roi_size[0]:
             center[0] = half_roi_size[0]
        else:
            center[0] = np.random.randint(half_roi_size[0], img_shape[0] - half_roi_size[0])
        center[1] = np.random.randint(half_roi_size[1], img_shape[1] - half_roi_size[1])
        center[2] = np.random.randint(half_roi_size[2], img_shape[2] - half_roi_size[2])

        labels_center = [int(i*s) for i,s in zip(center, self.scale_factor)]

        for key in self.keys:
            if "label" in key:
                data[key] = super().__call__(img=data[key], 
                                             slices=Crop.compute_slices(roi_center=labels_center, roi_size=self.roi_size_label), 
                                             lazy=lazy)
            else:
                data[key] = super().__call__(img=data[key], 
                                             slices=Crop.compute_slices(roi_center=center, roi_size=self.roi_size_image), 
                                             lazy=lazy)
        return data

def get_image_augmentation(config, split):
    # Rotation angles in radians
    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation_range']]

    # Calculate the initial crops size such that after rotation we don't lose any information
    inital_crop_size_in = get_inital_crop_size(config["input_size"])
    inital_crop_size_out = get_inital_crop_size(config["output_size"])
    print(f"Initial crop size: {inital_crop_size_in} -> {inital_crop_size_out}")
    if split == "train":
        # transforms = Compose([
        #     CustomCropRandomd(keys=['image', 'labels'], roi_size_image=[inital_crop_size_in for i in range(3)], roi_size_label=[inital_crop_size_out for i in range(3)]),
        #     ScaleIntensityRanged(keys=['image'],a_min=0, a_max=3, b_min=-1.0, b_max=1.0, clip=True),
        #     ScaleIntensityRanged(keys=['labels'],a_min=-1.5, a_max=1.5, b_min=-1.0, b_max=1.0, clip=True),
        #     RandRotated(keys=['image', 'labels'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
        #     RandZoomd(keys=['image', 'labels'], min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
        #     CenterSpatialCropd(keys=['image'], roi_size=config["input_size"]),
        #     CenterSpatialCropd(keys=['labels'], roi_size=config["output_size"]),
        #     RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
        #     RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
        #     RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
        #     RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
        #     RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
        #     RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
        #     RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
        #     RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
        #     ToTensord(keys=['image', 'labels'])
        # ])
         
        transforms = Compose([
            CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ScaleIntensityRanged(keys=['image'],a_min=0, a_max=3, b_min=-1.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=['labels'],a_min=-1.5, a_max=1.5, b_min=-1.0, b_max=1.0, clip=True),
            RandRotated(keys=['image', 'labels'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
            RandZoomd(keys=['image', 'labels'], min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
            # CenterSpatialCropd(keys=['image'], roi_size=config["input_size"]),
            # CenterSpatialCropd(keys=['labels'], roi_size=config["output_size"]),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
            RandFlipd(keys=['image', 'labels'], prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
            RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
            RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
            RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
            RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
            RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
            ToTensord(keys=['image', 'labels'])
        ])
    elif split == "val":
        transforms = Compose([
            # RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output_size"], random_size=False),
            CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ScaleIntensityRanged(keys=['image'],a_min=0, a_max=3, b_min=-1.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=['labels'],a_min=-1.5, a_max=1.5, b_min=-1.0, b_max=1.0, clip=True),
            ToTensord(keys=['image', 'labels'])
            ])
        print("VAL")
    elif split == "test":
                transforms = Compose([
                    ScaleIntensityRanged(keys=['image'],a_min=0, a_max=3, b_min=-1.0, b_max=1.0, clip=True),
                    ScaleIntensityRanged(keys=['labels'],a_min=-1.5, a_max=1.5, b_min=-1.0, b_max=1.0, clip=True),
                    # RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output_size"], random_size=False),
                    # CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
                    ToTensord(keys=['image', 'labels'])
                ])

    return transforms



def get_image_segmentation_augmentation(config, split):

    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation_range']]

    if split == "train":
         transforms = Compose([
            # RandSpatialCropd(keys=['image', 'mask', 'cortical', 'trabecular'], roi_size=config["input_size"], random_size=False, allow_missing_keys=True),
            ScaleIntensityRanged(keys=['image'],a_min=0, a_max=5, b_min=-1.0, b_max=1.0, clip=True),
            RandRotated(keys=['image', 'labels'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
            RandZoomd(keys=['image',  'mask', 'cortical', 'trabecular'], allow_missing_keys=True, min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
            # CenterSpatialCropd(keys=['image'], roi_size=config["input_size"]),
            # CenterSpatialCropd(keys=['labels'], roi_size=config["output_size"]),
            RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
            RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
            RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
            # RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
            # RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
            # RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
            # RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
            # RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
            ToTensord(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True)
        ])
    elif split == "test":
        transforms = Compose([
            ScaleIntensityRanged(keys=['image'],a_min=0, a_max=5, b_min=-1.0, b_max=1.0, clip=True),
            # RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output_size"], random_size=False),
            # CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ToTensord(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True)
        ])
    elif split == "val":
        transforms = Compose([
            # RandSpatialCropd(keys=['image', 'mask', 'cortical', 'trabecular'], roi_size=config["input_size"], random_size=False, allow_missing_keys=True),
            ScaleIntensityRanged(keys=['image'],a_min=0, a_max=5, b_min=-1.0, b_max=1.0, clip=True),
            ToTensord(keys=['image', 'mask', 'cortical', 'trabecular'], allow_missing_keys=True)
        ])

    crop = Compose([RandSpatialCropd(keys=['image', 'mask', 'cortical', 'trabecular'], roi_size=config["input_size"], random_size=False, allow_missing_keys=True)])
    identity = lambda x: x
    return transforms, identity if split == "test" else crop
              


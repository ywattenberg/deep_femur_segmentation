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
    HistogramNormalized,
    CopyItemsd,
    ThresholdIntensityd,
    GaussianSmoothd,
    Identityd,
)
import logging
from src.dataset.utils import get_inital_crop_size

class SizedCropRandomd(Randomizable, Crop):
    def __init__(self, keys, image_key, roi_size, mask_key=None, fraction_ones=0.1):
        """
        Args:
            keys: keys to be cropped
            roi_size: size of the cropped region
            mask_key: key of the mask to be used for cropping
            fraction_ones: minimum fraction of ones in the mask
        """
        super().__init__()
        self.roi_size_image = roi_size
        self.image_key = image_key
        self.keys = keys
        self.mask_key = mask_key
        self.fraction_ones = fraction_ones
    
    def __call__(self, data, lazy=False):
        img_shape = data[self.image_key].shape[-3:] 

        # Get Random Center Point for Crop
        half_roi_size = [int(i/2) for i in self.roi_size_image]
        
        center = self.get_center(img_shape)
        if self.mask_key is not None:
            while True:
                curr_shape = data[self.mask_key].shape[-3:] 
                scaled_center = [int(center[i] * curr_shape[i] / img_shape[i]) for i in range(3)]
                scaled_roi_size = [int(self.roi_size_image[i] * curr_shape[i] / img_shape[i]) for i in range(3)]
                new_mask = super().__call__(img=data[self.mask_key], 
                                            slices=Crop.compute_slices(roi_center=scaled_center, roi_size=scaled_roi_size), 
                                            lazy=lazy)
                if np.sum(new_mask) / np.prod(new_mask.shape[-3:]) > self.fraction_ones:
                    break
                else:
                    center = self.get_center(img_shape)
            

        for key in self.keys:
            curr_shape = data[key].shape[-3:] 
            scaled_center = [int (np.round(center[i] * curr_shape[i] / img_shape[i])) for i in range(3)]
            scaled_roi_size = [int(np.round(self.roi_size_image[i] * curr_shape[i] / img_shape[i])) for i in range(3)]
            data[key] = super().__call__(img=data[key], 
                                            slices=Crop.compute_slices(roi_center=scaled_center, roi_size=scaled_roi_size), 
                                            lazy=lazy)
        return data
    

    def get_center(self, img_shape):
        half_roi_size = [int(i/2) for i in self.roi_size_image]
        center = [0,0,0] # z,x,y

        if  half_roi_size[0] >= img_shape[0] - half_roi_size[0]:
             center[0] = half_roi_size[0]
        else:
            center[0] = np.random.randint(half_roi_size[0], img_shape[0] - half_roi_size[0]-1)
        center[1] = np.random.randint(half_roi_size[1], img_shape[1] - half_roi_size[1]-1)
        center[2] = np.random.randint(half_roi_size[2], img_shape[2] - half_roi_size[2]-1)
        return center
        

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
        pcct_intensity_scale = config["augmentation_params"]["pcct_intensity_scale"]
        hrpqct_intensity_scale = config["augmentation_params"]["hrpqct_intensity_scale"]
        transforms = Compose([
            SizedCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ScaleIntensityRanged(keys=['image'],a_min=pcct_intensity_scale[0], a_max=pcct_intensity_scale[1], b_min=pcct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
            ScaleIntensityRanged(keys=['labels'],a_min=hrpqct_intensity_scale[0], a_max=hrpqct_intensity_scale[1] , b_min=hrpqct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
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
            SizedCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ScaleIntensityRanged(keys=['image'],a_min=pcct_intensity_scale[0], a_max=pcct_intensity_scale[1], b_min=pcct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
            ScaleIntensityRanged(keys=['labels'],a_min=hrpqct_intensity_scale[0], a_max=hrpqct_intensity_scale[1] , b_min=hrpqct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
            ToTensord(keys=['image', 'labels'])
            ])
        print("VAL")
    elif split == "test":
                transforms = Compose([
                    ScaleIntensityRanged(keys=['image'],a_min=pcct_intensity_scale[0], a_max=pcct_intensity_scale[1], b_min=pcct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
                    ScaleIntensityRanged(keys=['labels'],a_min=hrpqct_intensity_scale[0], a_max=hrpqct_intensity_scale[1] , b_min=hrpqct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
                    # RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output_size"], random_size=False),
                    # CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
                    ToTensord(keys=['image', 'labels'])
                ])

    return transforms



def get_image_segmentation_augmentation(config, split):

    rotation_range = [i / 180 * np.pi for i in config['augmentation_params']['rotation_range']]
    pcct_intensity_scale = config["augmentation_params"]["pcct_intensity_scale"]
    hrpqct_intensity_scale = config["augmentation_params"]["hrpqct_intensity_scale"]

    default_transforms = [
        ScaleIntensityRanged(keys=['pcct'],a_min=pcct_intensity_scale[0], a_max=pcct_intensity_scale[1], b_min=pcct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
        ScaleIntensityRanged(keys=['image'],a_min=hrpqct_intensity_scale[0], a_max=hrpqct_intensity_scale[1] , b_min=hrpqct_intensity_scale[2], b_max=hrpqct_intensity_scale[3], clip=True),
    ]

    if split == "train":
        transforms = Compose(
            [SizedCropRandomd(keys=['image',  'mask', 'cortical', 'trabecular', 'pcct'], image_key='image',         roi_size=config["output_size"], mask_key='mask')]
              + default_transforms +
                [
                    RandRotated(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], range_x=rotation_range, range_y=rotation_range, range_z=rotation_range, prob=config["augmentation_params"]["p_rotation"]),
                    RandZoomd(keys=['image',  'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True, min_zoom=config["augmentation_params"]["min_zoom"], max_zoom=config["augmentation_params"]["max_zoom"], prob=config["augmentation_params"]["p_zoom"]),
                    # CenterSpatialCropd(keys=['image'], roi_size=config["input_size"]),
                    # CenterSpatialCropd(keys=['labels'], roi_size=config["output_size"]),
                    RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=0),
                    RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=1),
                    RandFlipd(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True, prob=config["augmentation_params"]["p_flip"], spatial_axis=2),
                    # RandGaussianNoised(keys=['image'], prob=config["augmentation_params"]["p_noise"], std=config["augmentation_params"]["noise_std"]),
                    # RandGaussianSmoothd(keys=['image'], prob=config["augmentation_params"]["p_smooth"], sigma_x=config["augmentation_params"]["smooth_sigma"], sigma_y=config["augmentation_params"]["smooth_sigma"], sigma_z=config["augmentation_params"]["smooth_sigma"]),
                    # RandScaleIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_scale"] ,factors=config["augmentation_params"]["intensity_scale_factors"]),
                    # RandShiftIntensityd(keys=['image'], prob=config["augmentation_params"]["p_intensity_shift"], offsets=config["augmentation_params"]["intensity_shift_offsets"]),
                    # RandAdjustContrastd(keys=['image'], prob=config["augmentation_params"]["p_contrast"], gamma=config["augmentation_params"]["contrast_gamma"]),
                    ToTensord(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True)
        ])
    elif split == "test":
        transforms = Compose(default_transforms + [
            # RandSpatialCropd(keys=['image', 'labels'], roi_size=config["output_size"], random_size=False),
            # CustomCropRandomd(keys=['image', 'labels'], roi_size_image=config["input_size"], roi_size_label=config["output_size"]),
            ToTensord(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True)
        ])
    elif split == "val":
        transforms = Compose(
            [SizedCropRandomd(keys=['image',  'mask', 'cortical', 'trabecular', 'pcct'], image_key='image',         roi_size=config["output_size"], mask_key='mask')]
            + default_transforms + 
            [ToTensord(keys=['image', 'mask', 'cortical', 'trabecular', 'pcct'], allow_missing_keys=True)
        ])

    return transforms
              


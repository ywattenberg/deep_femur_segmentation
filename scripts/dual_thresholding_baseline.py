import numpy as np
import SimpleITK as sitk
import os
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import skimage as ski
import matplotlib.pyplot as plt
import yaml
import torch

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.post_process import remove_islands_from_mask, keep_largest_connected_component_skimage, fill_in_gaps_in_mask
from src.dataset.dataset_segmentation import FemurSegmentationDataset

def extract_periosteal_surface(image, threshold):
    # Extract periosteal surface
    # Create a binary image of the periosteal surface
    periosteal_surface = ski.filters.median(image, footprint=np.ones((3,3,1))) > threshold
    
    # Median filter to remove noise
    # periosteal_surface = ski.filters.median(periosteal_surface, footprint=np.ones((3,3,1)))
    # Dialation 
    periosteal_surface = ski.morphology.binary_dilation(periosteal_surface, footprint=np.ones((15,15,1)))
    periosteal_surface = ski.morphology.binary_dilation(periosteal_surface, footprint=np.ones((7,7,1)))

    # connected components
    periosteal_surface = remove_islands_from_mask(periosteal_surface, 10)
    periosteal_surface = fill_in_gaps_in_mask(periosteal_surface, 10)

    # Erosion
    periosteal_surface = ski.morphology.binary_erosion(periosteal_surface, footprint=np.ones((15,15,1)))
    periosteal_surface = ski.morphology.binary_erosion(periosteal_surface, footprint=np.ones((7,7,1)))
    return periosteal_surface

def extract_endosteal_surface(image, periosteal_surface, threshold):
    # Extract endosteal surface
    # Create a binary image of the endosteal surface
    endosteal_surface = ski.filters.median(image, footprint=np.ones((3,3,1))) > threshold
    endosteal_surface = endosteal_surface < threshold


    endosteal_surface = endosteal_surface & periosteal_surface


    endosteal_surface = ski.morphology.binary_erosion(endosteal_surface, footprint=np.ones((8,8,1)))

    endosteal_surface = remove_islands_from_mask(endosteal_surface, 3)

    endosteal_surface = ski.morphology.binary_dilation(endosteal_surface, footprint=np.ones((8,8,1)))
    endosteal_surface = ski.morphology.binary_dilation(endosteal_surface, footprint=np.ones((10,10,1)))
    endosteal_surface = ski.morphology.binary_erosion(endosteal_surface, footprint=np.ones((15,15,1)))

    endosteal_surface = ski.filters.gaussian(endosteal_surface, sigma=3)
    # Thresholding
    endosteal_surface = endosteal_surface > 0.3
    return endosteal_surface


def main():
    config = yaml.load(open("config/segmentation_config.yaml", "r"), Loader=yaml.FullLoader)
    config["context_csv_path"] = r"HRpQCT_aim\numpy\Cropped_regions_test.csv"
    # Load the image
    dataset = FemurSegmentationDataset(config=config, split="test")
    for i in range(len(dataset)):
        print(i)
        image, hr, mask = dataset[i]
        image = image.squeeze(0)
        image = image.numpy().transpose((1, 2, 0))
        # Extract periosteal surface
        periosteal_surface = extract_periosteal_surface(image, 0.5)
        # plt.imshow(periosteal_surface[:,:,25], cmap="gray")
        # Extract endosteal surface
        endosteal_surface = extract_endosteal_surface(image, periosteal_surface, 0.7)
        periosteal_surface = np.bitwise_xor(periosteal_surface, endosteal_surface)
        # Save the surfaces
        periosteal_surface = periosteal_surface.transpose((2,0,1)).astype(np.uint8)
        endosteal_surface = endosteal_surface.transpose((2,0,1)).astype(np.uint8)
        
        image = image.transpose((2,0,1))
        image = sitk.GetImageFromArray(image)
        sitk.WriteImage(image, f"baseline/image_{i}.nii.gz")

        hr = hr.numpy()
        hr = sitk.GetImageFromArray(hr)
        hr.SetSpacing((0.5, 0.5, 0.5))
        sitk.WriteImage(hr, f"baseline/hr_{i}.nii.gz")

        periosteal_surface = sitk.GetImageFromArray(periosteal_surface)
        sitk.WriteImage(periosteal_surface, f"baseline/pred_mask_{i}_0.nii.gz")
        endosteal_surface = sitk.GetImageFromArray(endosteal_surface)
        sitk.WriteImage(endosteal_surface, f"baseline/pred_mask_{i}_1.nii.gz")
        cort_mask = mask[0].numpy().astype(np.uint8)
        trab_mask = mask[1].numpy().astype(np.uint8)
        cort_mask = sitk.GetImageFromArray(cort_mask)
        sitk.WriteImage(cort_mask, f"baseline/mask_{i}_0.nii.gz")
        trab_mask = sitk.GetImageFromArray(trab_mask)
        sitk.WriteImage(trab_mask, f"baseline/mask_{i}_1.nii.gz")

if __name__ == "__main__":
    main()



    
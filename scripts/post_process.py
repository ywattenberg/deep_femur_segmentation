import os 
import sys
import SimpleITK as sitk
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.post_process import postprocess_masks_iterative

def main(image_path, trab_mask_path, cort_mask_path, output_path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(trab_mask_path))
    cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(cort_mask_path))

    image = image.transpose(1, 2, 0)
    trab_mask = trab_mask.transpose(1, 2, 0)
    cort_mask = cort_mask.transpose(1, 2, 0)

    cort_mask, trab_mask = postprocess_masks_iterative(image, cort_mask, trab_mask, visualize=True)
    cort_mask = cort_mask.astype(np.uint8)
    trab_mask = trab_mask.astype(np.uint8)

    trab_mask = trab_mask.transpose(2, 0, 1)
    cort_mask = cort_mask.transpose(2, 0, 1)
    sitk.WriteImage(sitk.GetImageFromArray(cort_mask), os.path.join(output_path, "cort_mask.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(trab_mask), os.path.join(output_path, "trab_mask.nii.gz"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-t", "--trab_mask", required=True, help="Path to the trabecular mask")
    ap.add_argument("-c", "--cort_mask", required=True, help="Path to the cortical mask")
    ap.add_argument("-o", "--output", required=True, help="Path to the output directory")
    args = vars(ap.parse_args())
    main(args["image"], args["trab_mask"], args["cort_mask"], args["output"])


import os 
import sys
import SimpleITK as sitk
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.post_process import postprocess_masks_iterative, post_processing_retina

def process_sample(image, trab_mask, cort_mask, retina=False, visualize=False):
    if retina:
        print("Retina")
        cort_mask, trab_mask = post_processing_retina(image, cort_mask, trab_mask, visualize=visualize)
    else:
        cort_mask, trab_mask = postprocess_masks_iterative(image, cort_mask, trab_mask, visualize=visualize)
    cort_mask = cort_mask.astype(np.uint8)
    trab_mask = trab_mask.astype(np.uint8)
    return trab_mask, cort_mask

def main(image_path, trab_mask_path, cort_mask_path, output_path, folder, retina=False):
    if folder is not None:
        files = os.listdir(folder)
        for i in range(4):
            print(f"Processing sample {i}")
            image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, f"input_{i}.nii.gz")))
            trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, f"pred_mask_{i}_1.nii.gz")))
            cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, f"pred_mask_{i}_0.nii.gz")))

            image = image.transpose(1, 2, 0)
            trab_mask = trab_mask.transpose(1, 2, 0)
            cort_mask = cort_mask.transpose(1, 2, 0)

            trab_mask, cort_mask = process_sample(image, trab_mask, cort_mask, retina=retina, visualize=False)
            trab_name = f"pred_mask_{i}_1.nii.gz"
            cort_name = f"pred_mask_{i}_0.nii.gz"
            
            print(f"Output path: {output_path}")
            image = sitk.GetImageFromArray(image.transpose(2, 0, 1))
            sitk.WriteImage(image, os.path.join(output_path, f"input_{i}.nii.gz"))

            trab_mask = trab_mask.transpose(2, 0, 1)
            cort_mask = cort_mask.transpose(2, 0, 1)
            sitk.WriteImage(sitk.GetImageFromArray(cort_mask), os.path.join(output_path, cort_name))
            sitk.WriteImage(sitk.GetImageFromArray(trab_mask), os.path.join(output_path, trab_name))
            
    else:
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(trab_mask_path))
        cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(cort_mask_path))

        image = image.transpose(1, 2, 0)
        trab_mask = trab_mask.transpose(1, 2, 0)
        cort_mask = cort_mask.transpose(1, 2, 0)

        trab_mask, cort_mask = process_sample(image, trab_mask, cort_mask, retina=retina, visualize=True)
        trab_name = os.path.basename(trab_mask_path)
        cort_name = os.path.basename(cort_mask_path)

        trab_mask = trab_mask.transpose(2, 0, 1)
        cort_mask = cort_mask.transpose(2, 0, 1)
        image = sitk.GetImageFromArray(image.transpose(2, 0, 1))
        
        sitk.WriteImage(sitk.GetImageFromArray(cort_mask), os.path.join(output_path, cort_name))
        sitk.WriteImage(sitk.GetImageFromArray(trab_mask), os.path.join(output_path, trab_name))
        sitk.WriteImage(image, os.path.join(output_path, "input.nii.gz"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Path to the input image")
    ap.add_argument("-f", "--folder", required=False, help="Path to the folder containing the images")
    ap.add_argument("-t", "--trab_mask", required=False, help="Path to the trabecular mask")
    ap.add_argument("-c", "--cort_mask", required=False, help="Path to the cortical mask")
    ap.add_argument("-o", "--output", required=True, help="Path to the output directory")
    ap.add_argument("-r", "--retina", action="store_true", help="Flag to indicate if the input is a retina image")
    args = vars(ap.parse_args())
    main(args["image"], args["trab_mask"], args["cort_mask"], args["output"],  args["folder"], args["retina"])


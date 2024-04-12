import numpy as np
import os
import subprocess
import yaml
import zipfile
import io
import SimpleITK as sitk
import matplotlib.pyplot as plt


def main(path, roi):
    # # Find pcct file
    name = path.split("\\")[-1]
    print(f"Processing {name}")
    files = os.listdir(path)
    pcct_file = [file for file in files if file == "cutout_calibration_phantom.nii.gz"]
    # if len(pcct_file) == 0:
    #     print(f"Could not find pcct file in {path}")
    #     print(f"Name: {name}")
    #     return
    # pcct_file = pcct_file[0]
    # pcct_file = os.path.join(path, pcct_file)
    # print(f"Found pcct file: {pcct_file}")

    # # Read pcct file
    # image = sitk.ReadImage(pcct_file)
    # print(f"Image shape: {image.GetSize()}")
    # image = sitk.GetArrayFromImage(image)

    # min_value = np.min(image)
    # image = None
    # # Transform file
    # with open(os.path.join(path, "TransformParameters.0.txt"), "r") as f:
    #     # Find the line with the DefaultPixelValue value (DefaultPixelValue 0)
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         if "DefaultPixelValue 0" in line :
    #             lines[i] = f"(DefaultPixelValue {min_value})\n"
    #             print(f"Found line: {lines[i]} replacing with: {min_value}")
    #             break

    # with open(os.path.join(path, "TransformParameters.1.txt"), "w") as f:
    #     f.writelines(lines)

    # # Transform image
    # subprocess.run(["./elastix/transformix", "-in", pcct_file, "-out", path, "-tp", os.path.join(path, "TransformParameters.1.txt")])

    # Read transformed image
    image = sitk.ReadImage(os.path.join(path, "result.mha"))
    image = sitk.GetArrayFromImage(image)
    #image = image[:, roi[2]:roi[3], roi[0]:roi[1]]
    for i in range(0, image.shape[0], 250):
        plt.imshow(image[i, :, :,])
        plt.savefig(f"{name}_image_{i}.png")
    overall_mean = np.mean(image)
    overall_std = np.std(image)
    print(f"Overall mean: {overall_mean}, overall std: {overall_std}")


    # Normalize image
    image = (image - overall_mean) / overall_std
    overall_min = np.min(image)
    overall_max = np.max(image)
    overall_p99_5 = np.percentile(image, 99.5)
    overall_p00_5 = np.percentile(image, 0.5)
    print(f"Overall min: {overall_min}, overall max: {overall_max}, overall p99.5: {overall_p99_5}, overall p00.5: {overall_p00_5}")

    # Save statistics
    with open(os.path.join(path, "stats.yaml"), "w") as f:
        yaml.dump({"name": name, "mean": float(overall_mean), "std": float(overall_std), "min": float(overall_min), "max": float(overall_max), "p99_5": float(overall_p99_5), "p00_5": float(overall_p00_5)}, f)

    # Save image
    with zipfile.ZipFile(os.path.join(path, f"{name}.zip"), "w", zipfile.ZIP_DEFLATED) as f:
        for slice in range(image.shape[0]):
            io_buffer = io.BytesIO()
            np.save(io_buffer, image[slice, :, :])
            f.writestr(f"{slice}.npy", io_buffer.getvalue())
            io_buffer.close()
    
    # Calculate statistics



if __name__ == "__main__":
    image_folder_path = r"data\PCCT"

    folders = os.listdir(image_folder_path)
    
    dict = {"4":[299,858,392,1102], "6":[311,778,384,1110], "10":[377,795,411,1097], "11":[246,824,395,1087],"13":[370,800,309,954]}

    
    for key, roi in dict.items():
        sample = [folder for folder in folders if folder.startswith(f"{key}_")]
        if len(sample) > 0:
            sample = sample[0]
            main(os.path.join(image_folder_path, sample), roi)
    



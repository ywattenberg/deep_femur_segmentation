import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import yaml


def main(samples):
    df = pd.read_csv(r"data\HRpQCT_aim\Cropped_regions.csv", delimiter=";")
    for folder in os.listdir(r"data\PCCT"):
        files = os.listdir(os.path.join(r"data\PCCT", folder))
        print(f"Processing {folder.split('_')[0]}")
        if folder.split("_")[0] not in samples:
            print(f"Skipping {folder}")
            continue
        if "result.mha" not in files:
            with open("missing_pcct.txt", "a") as f:
                f.write(f"missing pcct {folder}\n")
            continue 
        
        image = sitk.ReadImage(os.path.join(r"data\PCCT", folder, "result.mha"))
        print(f"Image shape: {image.GetSize()}")

        name = "_".join(folder.split("_")[1:])
        print(f"Processing {name}")

        entries = df[df["SampName"].str.contains(name)]
        basic_path = os.path.join(r"data\HRpQCT_aim\numpy", f"{entries['SampName'].values[0].lower()}__")
        for i, entry in entries.iterrows():
            region = entry["Loc"]
            sample_folder = basic_path + region.lower()
            if not os.path.exists(sample_folder):
                print(f"Could not find {sample_folder}")
                with open("missing_pcct.txt", "a") as f:
                    f.write(f"missing pcct {sample_folder}\n")
                continue
            print(f"Sample folder: {sample_folder}")
            np_images = os.listdir(sample_folder)
            np_images = [image for image in np_images if "image" in image and image.endswith(".npy")]
            if len(np_images) == 0:
                print(f"Could not find np_images in {sample_folder}")
                with open("missing_pcct.txt", "a") as f:
                    f.write(f"np_images missing from {folder} region {region}\n")
                continue
            np_images = np_images[0]

            print(f"np_images: {os.path.join(sample_folder, np_images)}")
            np_images = np.load(os.path.join(sample_folder, np_images))
            size = np_images.shape
            size = [int(i/2) for i in size]
            minx = int(entry["MinX"]/2)
            miny = int(entry["MinY"]/2)
            minz = int(entry["MinZ"]/2)
            
            array = sitk.GetArrayFromImage(image[minx:minx+size[2], miny:miny+size[1], minz:minz+size[0]])
            print(f"Array shape: {array.shape}")
            mean = np.mean(array)
            std = np.std(array.astype(np.float32)).astype(np.float16)
            print(f"Mean: {mean}, std: {std}")
            array = array - mean
            array = array / std
            array = array.astype(np.float16)
            with open(os.path.join(sample_folder, "image_stats.yaml"), "r") as f:
                stats = yaml.safe_load(f)
            stats["pcct_mean"] = float(mean)
            stats["pcct_std"] = float(std)
            with open(os.path.join(sample_folder, "image_stats.yaml"), "w") as f:
                yaml.dump(stats, f)
            np.save(os.path.join(sample_folder, f"{name[:-2]}_pcct.npy"), array)





        


if __name__ == "__main__":
    main(['7'])
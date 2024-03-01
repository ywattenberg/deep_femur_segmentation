import os
import pandas as pd

def main():
    df = pd.read_csv(r"data\HRpQCT_aim\numpy\Cropped_regions_train.csv", delimiter=";")
    print(df)
    hr_folders = os.listdir(r"data\HR-pQCT")
    hr_folders = ["_".join(i.split("_")[1:]).lower() for i in hr_folders]
    hr_folders = [i for i in hr_folders if i.endswith("l") or i.endswith("r")]
    
    mask = df["SampName"].apply(lambda x: any([i in x.lower() for i in hr_folders]))
    df = df[mask]
    print(df)
    df.to_csv(r"data\HRpQCT_aim\numpy\Cropped_regions_train_filtered.csv", index=False, sep=";")

if __name__ == "__main__":
    main()

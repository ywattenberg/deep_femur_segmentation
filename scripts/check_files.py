import os
import argparse


def main():
    path1 = r"data\HRpQCT_annotated\pcct"
    files = os.listdir(path1)
    files = [f for f in files if os.path.isdir(os.path.join(path1, f))]

    path2 = r"data\HRpQCT_aim\numpy"
    files2 = os.listdir(path2)
    files2 = [f for f in files2 if os.path.isdir(os.path.join(path2, f))]

    missing = []
    for file in files2:
        print(os.path.join(path2, file))
        if len(os.listdir(os.path.join(path2, file))) < 4 and not file.endswith("h"):
            missing.append(file)

    print(missing)
    

if __name__ == "__main__":
    main()

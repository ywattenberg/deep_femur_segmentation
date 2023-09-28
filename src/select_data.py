import os
import shutil
import multiprocessing as mp

SOURCE_DIR = "C:\\Users\\Yannick\\Documents\\repos\\deep_femur_segmentation\\data\\PCCT\\1_2208_04835_R\\Ex_vivo_bone_0_20_Br89_Q3_R_10"
DESTINATION_DIR = "C:\\Users\\Yannick\\Documents\\repos\\deep_femur_segmentation\\data\\PCCT\\1_2208_04835_R\\truncated"
NUM_OF_FILES_TO_SKIP = 20

def copy_file(file):
    file_path = os.path.join(SOURCE_DIR, file)
    dest_path = os.path.join(DESTINATION_DIR, file)
    shutil.copyfile(file_path, dest_path)


def main():
    files = os.listdir(SOURCE_DIR)
    files.sort()
    print("Number of files: {}".format(len(files)))
    files = [f for f in files if (f.endswith(".dcm") or f.endswith(".DCM_1")) and int(f.split(".")[0].split("-")[-1]) % NUM_OF_FILES_TO_SKIP == 0]
    print("Number of files to copy: {}".format(len(files)))
    with mp.Pool(4) as pool:
        pool.map(copy_file, files)
    
    
if __name__ == "__main__":
    main()
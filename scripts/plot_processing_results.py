import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse



def plot_masks():
    folders = ["test copy", "test_2D copy", "test_retina copy"]
    fig, ax = plt.subplots(2, 5, figsize=(20, 9))
    ax[0][0].set_title("Input Image", fontsize=30)
    ax[0][1].set_title("Ground Truth", fontsize=30)
    ax[0][2].set_title("3D UNet", fontsize=30)
    ax[0][3].set_title("2D UNet", fontsize=30)
    ax[0][4].set_title("Retina UNet", fontsize=30)
    for i in range(2):
        image = sitk.GetArrayFromImage(sitk.ReadImage(f"data/{folders[0]}/input_{i}.nii.gz"))
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(f"data/{folders[0]}/mask_{i}_0.nii.gz"))
        pred_3d = sitk.GetArrayFromImage(sitk.ReadImage(f"data/{folders[0]}/pred_mask_{i}_0.nii.gz"))
        pred_2d = sitk.GetArrayFromImage(sitk.ReadImage(f"data/{folders[1]}/pred_mask_{i}_0.nii.gz"))
        pred_retina = sitk.GetArrayFromImage(sitk.ReadImage(f"data/{folders[2]}/pred_mask_{i}_0.nii.gz"))

        ax[i][0].imshow(image[0], cmap="gray")
        ax[i][0].axis("off")
        ax[i][1].imshow(ground_truth[0], cmap="gray")
        ax[i][1].axis("off")
        ax[i][2].imshow(pred_3d[0], cmap="gray")
        ax[i][2].axis("off")
        ax[i][3].imshow(pred_2d[0], cmap="gray")
        ax[i][3].axis("off")
        ax[i][4].imshow(pred_retina[0], cmap="gray")
        ax[i][4].axis("off")

    fig.tight_layout()
    plt.savefig("masks.png", dpi=400)
        


def main(pre_dir, post_dir):
    pre_dirl = os.listdir(pre_dir)
    post_dirl = os.listdir(post_dir)

    # ["Trab Dice", "Cort Dice", "Trab Jaccard", "Cort Jaccard", "Trab Hausdorff", "Cort Hausdorff"]
    fig, ax = plt.subplots(4, 5, figsize=(23, 20))
    ax[0][4].set_title("Trabecular Mask Post.", fontsize=30)
    ax[0][0].set_title("Input Image", fontsize=30)
    ax[0][1].set_title("Cortical Mask", fontsize=30)
    ax[0][2].set_title("Trabecular Mask", fontsize=30)
    ax[0][3].set_title("Cortical Mask Post.", fontsize=30)
    for i in range(4):
        cort_mask, trab_mask = sorted([mask for mask in pre_dirl if mask.startswith(f"pred_mask_{i}")])
        print(f"Preprocessing masks: {cort_mask}, {trab_mask}")
        cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pre_dir, cort_mask)))
        trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pre_dir, trab_mask)))
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pre_dir, f"input_{i}.nii.gz")))

        cort_mask_post, trab_mask_post = sorted([mask for mask in post_dirl if mask.startswith(f"pred_mask_{i}")])
        cort_mask_post = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(post_dir, cort_mask_post)))
        trab_mask_post = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(post_dir, trab_mask_post)))

        print(f"Shape of cort_mask: {cort_mask.shape}")
        print(f"Shape of trab_mask: {trab_mask.shape}")
        print(f"Shape of cort_mask_post: {cort_mask_post.shape}")
        print(f"Shape of trab_mask_post: {trab_mask_post.shape}")

        ax[i][0].imshow(image[0], cmap="gray")
        ax[i][0].axis("off")
        ax[i][1].imshow(cort_mask[0], cmap="gray")
        ax[i][1].axis("off")
        ax[i][2].imshow(trab_mask[0], cmap="gray")
        ax[i][2].axis("off")
        ax[i][3].imshow(cort_mask_post[0], cmap="gray")
        ax[i][3].axis("off")
        ax[i][4].imshow(trab_mask_post[0], cmap="gray")
        ax[i][4].axis("off")
    fig.tight_layout()
    plt.savefig("pre_post.png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the predicted masks")
    parser.add_argument("--pre_dir", type=str, help="Directory containing the preprocessed masks")
    parser.add_argument("--post_dir", type=str, help="Directory containing the postprocessed masks")
    parser.add_argument("--masks", "-m", action="store_true", help="Plot the masks")
    args = parser.parse_args()
    if args.masks:
        plot_masks()
    else:
        main(args.pre_dir, args.post_dir)



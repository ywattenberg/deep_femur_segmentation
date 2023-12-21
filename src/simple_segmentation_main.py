import torch
import yaml
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from trainer import Trainer
from dataset.dataset_segmentation import FemurSegmentationDataset 
from dataset.transforms import CustomCropRandomd

torch.set_default_dtype(torch.float32)

def main():
    config = yaml.safe_load(open("config/segmentation_config.yaml", "r"))
    if "seed" in config:
        torch.manual_seed(config["seed"])
    config["output_size"] = [1, 512, 512]
    dataset = FemurSegmentationDataset(config, split="test")
    model = monai_nets.UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=3 if config["use_cortical_and_trabecular"] else 1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    
    for i in range(len(dataset)):
        image, mask = dataset[i]
        axs, fig = plt.subplots(2,2)
        half_depth = int(image.shape[0]/2)
        fig[0,0].imshow(image[0,half_depth,:,:])
        fig[0,1].imshow(mask[0,half_depth,:,:])
        fig[1,0].imshow(mask[1,half_depth,:,:])
        fig[1,1].imshow(mask[2,half_depth,:,:])
        plt.show()


    # model = model.to("cuda")
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    # trainer = Trainer(model, dataset, DiceLoss(), optimizer, config)
    # trainer.train_test()

if __name__ == "__main__":
    main()




    

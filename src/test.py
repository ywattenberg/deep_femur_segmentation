from typing import Any
import torch
import monai.networks.nets as monai_nets
from dataset.dataset import FemurImageDataset 
import matplotlib.pyplot as plt
from trainer import Trainer

CONFIG = {
    "context csv path": r"C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data\test.csv",
    "base path": "",
    "use accelerator": False,
    "add channel dim": False,
    "output size": [64, 64, 64], # Default output size in form [z, x, y]
    "augmentation": True,
    "augmentation_params": {
        "rotation": [0, 360],
    }
}


def main():
    torch.set_float32_matmul_precision('medium')
    # Get Unet model from Monai
    model = monai_nets.basic_unet.BasicUNet(
            spatial_dims=3, in_channels=1, out_channels=1, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', 
            {'inplace': True, 'negative_slope': 0.1}), norm=('instance', {'affine': True}), bias=True, dropout=0.3, upsample='deconv')
    
    dataset = FemurImageDataset(config=CONFIG, split="val")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
        # split_test: float = None,
        # device=None,
        # batch_size=32,
        # epochs=10,
        # shuffle=True,
        # name=None,
        # test_metrics=None,
        # test_metric_names=None,
        # scheduler=None,
        # epochs_between_safe=1,
        # batches_between_safe=None,
        # split_random=False,
    trainer_config = { 
        "split_test": 0.2,
        "device": "cuda",
        "batch_size": 32,
        "epochs": 1,
        "shuffle": True,
        "name": "test",
        "test_metrics": [torch.nn.MSELoss()],
        "test_metric_names": ["MSE"],
        "epochs_between_safe": 1,
        "batches_between_safe": 20,
        "split_random": False,
        "tensorboard_path": "tensorboard"
    }


    trainer = Trainer(model, dataset, None, torch.nn.MSELoss(), optimizer, trainer_config)
    trainer.train_test()

    


    model.eval()
    for i in range(0, len(dataset), 100):
        x,y = dataset[i]
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(x[0].to("cpu").numpy()[0,:,:])
        ax[0,0].title.set_text("Input")
        ax[1,0] .imshow(y[0].to("cpu").numpy()[0,:,:])
        ax[1,0].title.set_text("Target")
        ax[1,1].imshow(model(x.unsqueeze(0))[0].to("cpu").detach().numpy()[0,:,:])
        plt.show()
    
        
   



if __name__ == "__main__":
    main()

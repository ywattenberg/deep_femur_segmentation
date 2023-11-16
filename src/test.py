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
    trainer = Trainer(model, dataset, test_data=None, split_test=0.2, loss_fn=torch.nn.MSELoss(), optimizer=optimizer, device="cuda", epochs=50, batches_between_safe=500, epochs_between_safe=2, name="test", test_metrics=[torch.nn.MSELoss()], test_metric_names=["MSE"], scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True))
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

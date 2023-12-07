from monai.networks.nets import basic_unet, unet
from monai.networks.blocks.convolutions import Convolution, ResidualUnit


from torch import nn

class BasicUNet(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(BasicUNet, self).__init__()
        assert type(config) == dict, "Config must be a dictionary."
        if "model" in config:
            self._config = config["model"]
        else:
            self._config = config
        self.UNet = basic_unet.BasicUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels, features=(32, 32, 64, 128, 256, 32), act=('LeakyReLU', 
            {'inplace': True, 'negative_slope': 0.1}), norm=('instance', {'affine': True}), bias=True, dropout=0.3, upsample='deconv'
        )


    def forward(self, x):
        return self.model(x)


class UpsampleBasicUNet(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(UpsampleBasicUNet, self).__init__()
        act = ('LeakyReLU', 
            {'inplace': True, 'negative_slope': 0.1})
        norm = ('instance', {'affine': True})
        self.UNet = basic_unet.BasicUNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels, features=(32, 32, 64, 128, 256, 32), act=act, norm=norm, bias=True, dropout=0.3, upsample='deconv'
        )
        self.upcat_0 = basic_unet.UpCat(
            spatial_dims=3, in_chns=32, cat_chns=0, out_chns=32, act=act, norm=norm, bias=True, dropout=0.3, upsample='deconv'
        )

        self.final_conv = basic_unet.Conv["conv", 3](32, out_channels, kernel_size=1)


    def forward(self, x):
        x0 = self.UNet.conv_0(x)
        x1 = self.UNet.down_1(x0)
        x2 = self.UNet.down_2(x1)
        x3 = self.UNet.down_3(x2)
        x4 = self.UNet.down_4(x3)

        u4 = self.UNet.upcat_4(x4, x3)
        u3 = self.UNet.upcat_3(u4, x2)
        u2 = self.UNet.upcat_2(u3, x1)
        u1 = self.UNet.upcat_1(u2, x0)
        u0 = self.upcat_0(u1, None)

        return self.final_conv(u0)
    

class UpsampleUNet(nn.Module):
    def __init__(self, in_channels, out_channels, config) -> None:
        super().__init__()
        self._config = config
        if "model" in config:
            self._config = config["model"]
        self.UNet = unet.UNet(
            spatial_dims=self._config["spatial_dims"],
            in_channels=in_channels,
            out_channels=out_channels,
            channels=self._config["features"],
            strides=self._config["strides"],
            dropout=0.3,
            norm=self._config["norm"],
            act=self._config["activation"],
        )
        l_conv = self.UNet.model[-1]
        self.UNet.model[-1] = Convolution(
            spatial_dims=self._config["spatial_dims"],
            in_channels=l_conv.in_channels,
            out_channels=self._config["features"][0],
            strides=[self._config["strides"][-1] for _ in range(self._config["spatial_dims"])],
            kernel_size=[3 for _ in range(self._config["spatial_dims"])],
            act=self._config["activation"],
            norm=self._config["norm"],
            dropout=self._config["dropout"],
            bias=self._config["bias"],
            is_transposed=True,
        )

        self.UNet.model.append(
            Convolution(
                spatial_dims=self._config["spatial_dims"],
                in_channels=self._config["features"][0],
                out_channels=out_channels,
                strides=[self._config["strides"][-1] for _ in range(self._config["spatial_dims"])],
                kernel_size=[3 for _ in range(self._config["spatial_dims"])],
                act=self._config["activation"],
                norm=self._config["norm"],
                dropout=self._config["dropout"],
                bias=self._config["bias"],
                conv_only=True,
                is_transposed=True,
            )
        )

        self.model = self.UNet.model

    def forward(self, x):
        return self.UNet(x)


            









        
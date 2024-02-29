from monai.networks.nets import basic_unet, unet
from monai.networks.blocks.convolutions import Convolution, ResidualUnit

import torch
from torch import nn


class Retina_UNet(unet.UNet):
    def __init__(self, in_channels, out_channels_mask, out_channels_upsample, config) -> None:
        self._config = config
        if "model" in config:
            self._config = config["model"]
        super
        super().__init__(
            spatial_dims=self._config["spatial_dims"],
            in_channels=in_channels,
            out_channels=out_channels_mask,
            channels=self._config["features"][1:],
            strides=self._config["strides"][1:],
            dropout=self._config["dropout"],
            norm=self._config["norm"],
            act=self._config["activation"],
        )

        def _create_block(
            inc: int, outc: int, channels, strides, is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(self._config["features"][0],self._config["features"][0],self._config["features"][1:], self._config["strides"][1:] , False)
        down = self._get_down_layer(in_channels, self._config["features"][0], 2, False)
        up = self._get_up_layer(self._config["features"][0]*2, self._config["upsample_features"], 2, False)
        self.model = self._get_connection_block(down, up, self.model)
        self.upsample_head = self._get_up_layer(self._config["upsample_features"], out_channels_upsample, 2, True)
        self.mask_head = self._get_up_layer(self._config["upsample_features"], out_channels_mask, 1, True)

        
    def forward(self, input):
        x = self.model(input)
        mask = self.mask_head(x)
        upsample_image = self.upsample_head(x)
        return [mask, upsample_image]
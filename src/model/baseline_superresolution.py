import torch
import torch.nn as nn

class SimpleUpsampler(nn.Module):
    def __init__(self, output_size, mode="bilinear",  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Upsample(size=self.output_size, mode=mode, align_corners=True),
        )

    def forward(self, x):
        return self.model(x)
    
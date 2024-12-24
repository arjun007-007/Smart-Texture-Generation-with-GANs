import torch
import torch.nn as nn
import torch.nn.functional as F


class Blur(nn.Module):
    """
    Same blur as used in the generator for consistency.
    """
    def __init__(self, kernel=(1, 3, 3, 1)):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[None, None, :] * kernel[None, :, None]  # Outer product
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel.expand(x.shape[1], 1, *self.kernel.shape),
                        padding=int((self.kernel.shape[-1] - 1) / 2),
                        groups=x.shape[1])


def downsample(x):
    return F.avg_pool2d(x, 2)


class DiscriminatorBlock(nn.Module):
    """
    Discriminator block that progressively downsamples.
    """
    def __init__(self, in_channels, out_channels, blur_kernel=(1,3,3,1)):
        super().__init__()
        self.blur = Blur(blur_kernel)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.blur(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = downsample(x)
        return x


class Discriminator(nn.Module):
    """
    Simplified StyleGAN2 Discriminator.
    """
    def __init__(self, channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.from_rgb = nn.Conv2d(3, channels[0], 1)

        # Build a chain of blocks that scale up in channels as resolution goes down
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(DiscriminatorBlock(channels[i], channels[i+1]))
        self.blocks = nn.Sequential(*blocks)

        # Final layers to get a single real/fake score
        self.final_conv = nn.Conv2d(channels[-1], channels[-1], 4)
        self.final_linear = nn.Linear(channels[-1], 1)

    def forward(self, x):
        """
        x: (batch_size, 3, H, W)
        Returns: (batch_size, 1) real/fake score
        """
        out = self.from_rgb(x)
        out = self.blocks(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        return out

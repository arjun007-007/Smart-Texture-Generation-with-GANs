import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Blur(nn.Module):
    """
    A simple blur layer (anti-aliasing) that helps reduce checkerboard artifacts
    caused by transposed convolutions.
    """
    def __init__(self, kernel=(1, 3, 3, 1)):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[None, None, :] * kernel[None, :, None]  # Outer product
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        # Group conv for applying the blur kernel
        return F.conv2d(x, self.kernel.expand(x.shape[1], 1, *self.kernel.shape),
                        padding=int((self.kernel.shape[-1] - 1) / 2),
                        groups=x.shape[1])


class StyledConv(nn.Module):
    """
    A style-modulated convolution layer with optional noise injection.
    This is a simplified version of the StyleGAN2 styled convolution block.
    """
    def __init__(self, in_channels, out_channels, kernel_size, style_dim,
                 upsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.blur = Blur(blur_kernel)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size//2)
        self.scale_conv = nn.Linear(style_dim, in_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Optional: learnable noise scaling
        self.noise_scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style, noise=None):
        batch_size, _, height, width = x.shape

        # 1) Upsample + Blur
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self.blur(x)

        # 2) Style scaling
        style_scale = self.scale_conv(style).unsqueeze(2).unsqueeze(3)
        # shape: (batch_size, in_channels, 1, 1)
        x = x * (style_scale + 1)

        # 3) Convolution
        x = self.conv(x)

        # 4) Add noise
        if noise is None:
            noise = torch.randn(batch_size, 1, height * (2 if self.upsample else 1),
                                width * (2 if self.upsample else 1),
                                device=x.device)
        x = x + self.noise_scale * noise

        # 5) Activation
        x = self.activation(x)

        return x


class ToRGB(nn.Module):
    """
    Convert final features to RGB. Each layer can optionally add to the final output.
    """
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, 1)
        self.scale_conv = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        style_scale = self.scale_conv(style).unsqueeze(2).unsqueeze(3)
        out = self.conv(x * (style_scale + 1))
        return out


class Generator(nn.Module):
    """
    Simplified StyleGAN2 Generator. For full usage, you'd also implement a MappingNetwork
    that outputs per-layer style codes. Here we assume 'style' is already provided.
    """
    def __init__(self, style_dim=512, channels=[512, 256, 128, 64, 32], blur_kernel=(1,3,3,1)):
        super().__init__()
        self.style_dim = style_dim

        # Initial constant input (4x4)
        self.const_input = nn.Parameter(torch.randn(1, channels[0], 4, 4))

        # Create a list of styled blocks
        self.layers = nn.ModuleList()
        in_channels = channels[0]
        for out_channels in channels:
            self.layers.append(
                StyledConv(in_channels, out_channels, 3, style_dim,
                           upsample=(out_channels != channels[0]),
                           blur_kernel=blur_kernel)
            )
            in_channels = out_channels

        # ToRGB layer for the final output
        self.to_rgb = ToRGB(channels[-1], style_dim)

    def forward(self, style, noise=None):
        """
        style: (batch_size, style_dim)
        noise: optional noise injection at each layer.
        If noise is None, random noise is generated on the fly.
        """
        batch_size = style.shape[0]
        x = self.const_input.expand(batch_size, -1, -1, -1)

        for layer in self.layers:
            x = layer(x, style, noise=noise)

        out = self.to_rgb(x, style)
        return out

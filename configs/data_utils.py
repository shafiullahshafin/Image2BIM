"""
Data and positional encoding utilities for HEAT model.
"""

import math
import numpy as np
import torch


def positional_encoding_2d(d_model: int, height: int, width: int) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional encoding matrix of shape (d_model, height, width).
    """
    if d_model % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model})")
    pe = torch.zeros(d_model, height, width)
    d_half = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_half, 2) * -(math.log(10000.0) / d_half))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_half + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe


def positional_encoding_1d(d_model: int, length: int) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional encoding matrix of shape (length, d_model).
    """
    if d_model % 2 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})")
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def get_pixel_features(image_size: int = 256, d_pe: int = 128):
    """
    Generate 2D pixel coordinates and corresponding positional encoding features.
    """
    all_pe = positional_encoding_2d(d_pe, image_size, image_size)
    pixels_x = np.arange(0, image_size)
    pixels_y = np.arange(0, image_size)

    xv, yv = np.meshgrid(pixels_x, pixels_y)
    all_pixels = list()
    for i in range(xv.shape[0]):
        pixs = np.stack([xv[i], yv[i]], axis=-1)
        all_pixels.append(pixs)
    pixels = np.stack(all_pixels, axis=0)

    pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
    pixel_features = pixel_features.permute(1, 2, 0)
    return pixels, pixel_features

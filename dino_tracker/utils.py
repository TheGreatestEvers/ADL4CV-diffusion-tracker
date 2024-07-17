import gc
import torch
from einops import rearrange
import torchvision.transforms as T

import os


def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=False, normalize_w=False, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] / (t - 1)  # normalize to [0,1]
        samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values
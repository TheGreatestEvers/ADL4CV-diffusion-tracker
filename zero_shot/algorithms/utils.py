import torch
from torchvision.io import read_video

def load_video(
        video_path: str
) -> torch.tensor:
    
    """
    Read singel video instance from file

    Args:
        video_path: path to video file

    Return:
        video tensor: (FxCxHxW)
    """

    video, _, _ = read_video(video_path, pts_unit='sec')

    return video
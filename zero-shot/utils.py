import torch
from torchvision.io import read_video

def load_video(
        video_path: str
) -> torch.tensor:
    video, _, _ = read_video(video_path, pts_unit='sec', output_format='TCHW')

    return video
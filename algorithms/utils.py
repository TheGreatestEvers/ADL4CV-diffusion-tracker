import os
import yaml
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

def read_config_file(config_file):
    '''
    Reads an configuration file in yaml format and returns the dictionary
    '''
    assert os.path.exists(config_file), f"{config_file} not found"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
        
    return config

def feature_collate_fn(batch):
    """Custom collate function that returns the input batch as is, which is a list of dictionaries."""
    return batch

from PIL import Image
import os
import torch
import subprocess
from torch.utils.data import DataLoader

from algorithms.utils import read_config_file, feature_collate_fn
from algorithms.feature_extraction_loading import FeatureDataset


def save_video_frames(video, output_folder):
    """
    Saves frames from a video tensor as image files.
    
    Parameters:
    - video_tensor (torch.Tensor): The video tensor of shape (num_frames, height, width, channels).
    - output_folder (str): The folder where the image files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    num_frames = video.shape[0]

    for i in range(num_frames):
        # Get the frame
        frame = video[i]

        # Convert the frame to a PIL Image
        frame_image = Image.fromarray(frame)

        # Create the filename
        filename = f"{i:05d}.png"

        # Save the image
        frame_image.save(os.path.join(output_folder, filename))


if __name__ == "__main__":
    config = read_config_file("configs/config.yaml")
    dataset = FeatureDataset(feature_dataset_path=config['dataset_dir'])

    dataloader = DataLoader(dataset, collate_fn=feature_collate_fn)

    for i, data in enumerate(dataloader):
        data = data[0]

        video_folder = os.path.join('a_video_dir', 'video_' + str(i), 'video')

        os.makedirs(video_folder, exist_ok=True)

        save_video_frames(data['video'][0], video_folder)

        feature_path = os.path.join(config['dataset_dir'], 'video_' + str(i) + '.pkl')

        trajectory_path = os.path.join('a_video_dir', 'video_' + str(i), 'of_trajectories.pt')

        mask_path = os.path.join('a_video_dir', 'video_' + str(i), 'mask')

        subprocess.run(["python", "extract_trajectories.py", "--frames-path", video_folder, "--output-path", trajectory_path, "--feature-path", feature_path, "--mask-path", mask_path])

        if i >= 0:
            break
import os
import torch
import pickle
import copy
import torchvision.transforms.functional as functional
from torch.utils.data import Dataset

from algorithms.diffusion_wrapper import DiffusionWrapper
from evaluation.evaluation_datasets import create_davis_dataset

def feature_collate_fn(batch):
    """Custom collate function that returns the input batch as is, which is a list of dictionaries."""
    return batch

class FeatureDataset(Dataset):
    """
    Dataset for handling video features. Each item in the dataset is a dictionary containing:
    - 'video': The video data.
    - 'query_points': Query points.
    - 'target_points': Ground truth points.
    - 'occluded': Information about occluded points in the video.
    - 'trackgroup': Group tracking information.
    - 'features': Extracted video diffusion features of the video.
    """
    def __init__(self, feature_dataset_path: str = 'output/features/'):

        self.datasets = []

        for file in os.listdir(feature_dataset_path):
            if not file.endswith(".pkl"):
                continue
            
            with open(os.path.join(feature_dataset_path, file), 'rb') as dataset_file:
                dataset = pickle.load(dataset_file)
                self.datasets.extend(dataset)

                dataset_file.close()

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]

def extract_diffusion_features(input_dataset_paths: dict, output_dataset_path: str = 'output/features/', diffustion_model_path: str = './text-to-video-ms-1.7b'):
    """
    Extract and save video diffusion features from input datasets.

    Parameters:
    - input_dataset_paths: Dictionary where keys are dataset names and values are paths to the datasets.
    - output_dataset_path: Directory path where the output .pkl files with features will be saved.
    - diffusion_model_path: Path to the pre-trained diffusion model.
    """

    datasets = {}
    
    diffusion_wrapper = DiffusionWrapper(diffustion_model_path)
    
    if 'davis' in input_dataset_paths.keys():
        datasets['davis'] = create_davis_dataset(input_dataset_paths['davis'])

    for dataset_name, dataset_values in datasets.items():
        with open(os.path.join(output_dataset_path, dataset_name + '.pkl'), 'wb') as dataset_feature_file:
            dataset_with_features = []

            for data in dataset_values:
                data_with_features_dict = data[dataset_name]

                video_tensor = torch.tensor(data_with_features_dict['video'])
                video_features_dict = diffusion_wrapper.extract_video_features(video_tensor, "")

                data_with_features_dict['features'] = copy.deepcopy(video_features_dict)
                dataset_with_features.append(data_with_features_dict)
            
            pickle.dump(dataset_with_features, dataset_feature_file)

            dataset_feature_file.close()

def concatenate_video_features(features):
    """
    Concatenates video feature tensors after resizing them to a uniform size.

    Args:
        features: A dictionary containing feature maps. The dictionary keys can be considered as feature names
                        and the values are lists of feature tensors.

    Returns:
        torch.Tensor: A single concatenated feature map tensor of shape (BxFxCfxHxW)
    """

    max_height_width = max(ft.shape[-1] for fts in features.values() for ft in fts)

    feature_map = torch.cat([functional.resize(ft, [max_height_width] * 2) for fts in features.values() for ft in fts], dim=1)
    
    return feature_map


if __name__ == '__main__':
    extract_diffusion_features(input_dataset_paths={'davis': '/tapvid_davis/tapvid_davis.pkl'}, diffustion_model_path='../text-to-video-ms-1.7b/')
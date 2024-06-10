import os
import torch
import pickle
import copy
import gc
import torchvision.transforms.functional as functional
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as torchfunc

from algorithms.diffusion_wrapper import DiffusionWrapper
from evaluation.evaluation_datasets import create_davis_dataset

from sklearn.decomposition import PCA

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

        self.dataset_cnt = 0
        self.feature_dataset_path = feature_dataset_path

        for file in os.listdir(feature_dataset_path):
            if file.endswith(".pkl"):
                self.dataset_cnt += 1

    def __len__(self):
        return self.dataset_cnt

    def __getitem__(self, idx):
        dataset = {}

        with open(os.path.join(self.feature_dataset_path, 'video_' + str(idx) + '.pkl'), 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)

        return dataset
    
def do_pca(feature_tensor: torch.Tensor, n_components: int):
    F, C, H, W = feature_tensor.shape

    # Flatten the spatial dimensions, but keep the channel dimension
    flattened_features = feature_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()

    # Perform PCA on the channel dimension
    pca = PCA(n_components=n_components)
    transformed_features = pca.fit_transform(flattened_features)

    # Reshape back to the original structure, but with the new number of components
    transformed_features = torch.from_numpy(transformed_features).view(F, H, W, n_components)

    # Permute to match the original tensor format (F, C, H, W)
    transformed_features = transformed_features.permute(0, 3, 1, 2)

    return transformed_features

def do_pooling(feature_tensor: torch.Tensor, kernel_size= 4, stride=4, mode="max"):
    """
    Perform pooling along channel dimension of feature_tensor
    """
    F, C, H, W = feature_tensor.shape

    if mode != "max" and mode != "avg":
        ValueError("Mode has to either be max or avg.")

    # Reshape to apply max pooling over the channel dimension
    feat_tensor_reshaped = feature_tensor.permute(0, 2, 3, 1).contiguous()  # shape: F x H x W x C
    feat_tensor_reshaped = feat_tensor_reshaped.view(-1, C)  # shape: (F*H*W) x C

    # Calculate necessary padding
    padding_needed = (stride - (C % stride)) % stride

    if mode == "max":
        # Apply 1D max pooling
        tensor_pooled = torchfunc.max_pool1d(feat_tensor_reshaped.unsqueeze(1), kernel_size=kernel_size, stride=stride, padding=padding_needed).squeeze(1)  # shape: (F*H*W) x (C//stride)
    else:
        # Apply 1D avg pooling
        tensor_pooled = torchfunc.avg_pool1d(feat_tensor_reshaped.unsqueeze(1), kernel_size=kernel_size, stride=stride, padding=padding_needed).squeeze(1)  # shape: (F*H*W) x (C//stride)

    # Reshape the tensor back to the original spatial dimensions
    C_new = tensor_pooled.shape[1]
    tensor_pooled = tensor_pooled.view(F, H, W, C_new)  # shape: F x H x W x (C//stride)
    tensor_pooled = tensor_pooled.permute(0, 3, 1, 2).contiguous()  # shape: F x (C//stride) x H x W

    return tensor_pooled

    
def restrict_frame_size_to(video_feature_tensor: torch.Tensor, max_frame_size: int = 2 ** 20):
    F, C, H, W = video_feature_tensor.shape

    reduction_factor = max_frame_size / (C * H * W)

    if reduction_factor < 1:
        C_n = int(reduction_factor * C)
        video_feature_tensor = do_pca(video_feature_tensor, n_components=C_n)

    return video_feature_tensor

def extract_diffusion_features(
        input_dataset_paths: dict, 
        output_dataset_path: str = 'output/features/', 
        diffusion_model_path: str = './text-to-video-ms-1.7b', 
        restrict_frame_size: bool = False,
        restrict_ncomponents: bool = False,
        max_frame_size: int = 2 ** 20,
        n_components: int = 10,
        enable_vae_slicing: bool = True,
        use_decoder_features: bool = True
        ):
    """
    Extract and save video diffusion features from input datasets.

    Parameters:
    - input_dataset_paths (dict): A dictionary where keys are dataset names and values are paths to the datasets.
    - output_dataset_path (str): Directory path where the output .pkl files with extracted features will be saved. Defaults to 'output/features/'.
    - diffusion_model_path (str): Path to the pre-trained diffusion model. Defaults to './text-to-video-ms-1.7b'.
    - restrict_frame_size (bool): If True, restricts the size of video frames. Defaults to False.
    - restrict_ncomponents (bool): If True, applies PCA to reduce the number of components in the features. Defaults to False.
    - max_frame_size (int): Maximum allowed size for video frames when restrict_frame_size is True. Defaults to 2**20.
    - n_components (int): Number of components to keep when applying PCA if restrict_ncomponents is True. Defaults to 10.
    - enable_vae_slicing (bool): If True, enables slicing in the Variational Autoencoder (VAE) for memory efficiency. Defaults to True.
    - use_decoder_features (bool): If True, uses features from the decoder part of the diffusion model. Defaults to True.
    """

    datasets = {}
    
    diffusion_wrapper = DiffusionWrapper(diffusion_model_path, enable_vae_slicing=enable_vae_slicing)
    
    if 'davis' in input_dataset_paths.keys():
        datasets['davis'] = create_davis_dataset(input_dataset_paths['davis'])

    for dataset_name, dataset_values in datasets.items():
        print('Dataset: ' + dataset_name)

        os.makedirs(os.path.join(output_dataset_path, dataset_name), exist_ok=True)

        for data_idx, data in tqdm(enumerate(dataset_values), desc="Progress: "):
            data_with_features_dict = data[dataset_name]

            prompt = ""
            with open(os.path.join('../tapvid_davis_prompts/', 'prompt_' + f'{data_idx:03}' + '.txt'), 'r') as prompt_file:
                prompt = prompt_file.read()

                prompt_file.close()
            print(prompt)

            video_tensor = torch.tensor(data_with_features_dict['video'])
            video_features_dict = diffusion_wrapper.extract_video_features(video_tensor, prompt = prompt, use_decoder_features=use_decoder_features)

            if restrict_frame_size or restrict_ncomponents:
                for vfk, vfvs in video_features_dict.items():
                    for idx, vfv in enumerate(vfvs):
                        video_features_dict[vfk][idx] = restrict_frame_size_to(vfv, max_frame_size) if restrict_frame_size else do_pca(vfv, n_components)
                    
            data_with_features_dict['features'] = copy.deepcopy(video_features_dict)

            with open(os.path.join(output_dataset_path, dataset_name, 'video_' + str(data_idx)) + '.pkl', 'wb') as data_with_features_dict_file:
                pickle.dump(data_with_features_dict, data_with_features_dict_file)

                data_with_features_dict_file.close()

            del data_with_features_dict, video_features_dict
            gc.collect()

def concatenate_video_features(features, force_max_spatial: int = None, perform_pca: bool = False, n_components: int = 10, perform_pooling: bool = False):
    """
    Concatenates video feature tensors after resizing them to a uniform size.

    Args:
        features: A dictionary containing feature maps. The dictionary keys can be considered as feature names
                        and the values are lists of feature tensors.

    Returns:
        torch.Tensor: A single concatenated feature map tensor of shape (BxFxCfxHxW)
    """

    if perform_pca == True and perform_pooling == True:
        ValueError("Only either pooling or pca allowed.")

    feature_maps = []

    if force_max_spatial is None:
        max_height_width = max(ft.shape[-1] for fts in features.values() for ft in fts)
    else:
        max_height_width = force_max_spatial

    if perform_pca:
        for fts in features.values():
            for ft in fts:
                feature_maps.append(do_pca(ft, n_components))

        feature_map = torch.cat([functional.resize(ft, [max_height_width] * 2) for ft in feature_maps], dim=1)
    elif perform_pooling:
        for fts in features.values():
            for ft in fts:
                feature_maps.append(do_pooling(ft, mode="max"))

        feature_map = torch.cat([functional.resize(ft, [max_height_width] * 2) for ft in feature_maps], dim=1)
    else:
        feature_map = torch.cat([functional.resize(ft, [max_height_width] * 2) for fts in features.values() for ft in fts], dim=1)
    
    return feature_map


if __name__ == '__main__':
    extract_diffusion_features(input_dataset_paths={'davis': '/tapvid_davis/tapvid_davis.pkl'}, diffusion_model_path='../text-to-video-ms-1.7b/')
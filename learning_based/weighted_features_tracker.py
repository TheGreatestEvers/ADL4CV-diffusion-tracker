import torch
from torchvision.transforms.functional import resize 
from algorithms.heatmap_generator import HeatmapGenerator
from algorithms.zero_shot_tracker import ZeroShotTracker
from algorithms.feature_extraction_loading import concatenate_video_features

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

class WeightedFeaturesTracker(torch.nn.Module):
    """
    Implements a tracker which weights all feature blocks individually before calculating the heatmap.
    """

    def __init__(self, feature_dict) -> None:
        super().__init__()

        self.heatmap_generator = HeatmapGenerator()
        self.tracker = ZeroShotTracker()

        # Create parameter dict
        self.params = torch.nn.ParameterDict()

        for block_name, block_feature_list in feature_dict.items():
            self.params[block_name] = torch.nn.Parameter(torch.ones(len(block_feature_list)))



    def forward(self, feature_dict, query_points):
        """
        Args:
            feature_dict: Dictionary containing feature tensors
            query_points: Tensor of query points to track. Dimensions: [N, 3]
        
        Returns:
            Tensor of tracking results. Dimension: [N, F, 2]
        """

        # Scale each feature block with respective weight
        for block_name, block_feature_list in feature_dict.items():
            feature_dict[block_name] = [weight * feature_maps for weight, feature_maps in zip(self.params[block_name], block_feature_list)]

        # Concat all feature maps
        concat_features = concatenate_video_features(feature_dict)

        # Calculate heatmaps
        hmps = self.heatmap_generator.generate(concat_features, query_points)

        # Tracking
        tracks = self.tracker.track(hmps)

        return tracks


class WeightedHeatmapsTracker(torch.nn.Module):
    """
    Implements a tracker which calculates a seperate heatmap for each feature block and then sums up the weighted heatmaps.
    """

    def __init__(self, feature_dict) -> None:
        super().__init__()

        self.heatmap_generator = HeatmapGenerator()
        self.tracker = ZeroShotTracker()

        # Create parameter dict
        self.params = torch.nn.ParameterDict()

        for block_name, block_feature_list in feature_dict.items():
            self.params[block_name] = torch.nn.Parameter(torch.ones(len(block_feature_list)))



    def forward(self, feature_dict, query_points):
        """
        Calculate seperate heatmap for each feature block and sum them together in weighted fashion.

        Args:
            feature_dict: Dictionary containing feature tensors
            query_points: Tensor of query points to track. Dimensions: [N, 3]
        
        Returns:
            Tensor of tracking results. Dimension: [N, F, 2]
        """

        hmps_list = []

        for block_name, block_feature_list in feature_dict.items():
            for i, block_features in enumerate(block_feature_list):
                resized_features = resize(block_features,  (256, 256))

                hmps = self.heatmap_generator.generate(resized_features, query_points)

                # Apply weight to heatmap
                hmps *= self.params[block_name][i]

                hmps_list.append(hmps)
            
        # Sum up to single heatmap
        concat_heatmaps = torch.sum(torch.stack(hmps_list), dim=0)

        tracks = self.tracker.track(concat_heatmaps)

        return tracks
        

            






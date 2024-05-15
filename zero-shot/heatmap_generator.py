import torch
import numpy as np
import math

class HeatmapGenerator:
    """
    Used for generating Heatmaps
    """

    def __init__(self) -> None:
        ...
    
    def generate(self, feature_maps, target_coordinates):
        """
        Generates Heatmaps.

        Args:
            features_maps: Tensor with concatenation of feature maps. Dimension: [Frames, Height, Width, Channels]
            target_coordinates: Tuple with coordinates of point to track. Format: (y, x, time)

        Returns:
            heatmaps: Tensor of Heatmaps. Dimension: [Frames, Height, Width]
        """
        
        point_proj = self.__project_point_coordinates(target_coordinates)
        print("projected point: ")
        print(point_proj)

        target_feat_vec = self.__get_feature_vec_bilinear(feature_maps, point_proj)
        print("sample feat vec")
        print(target_feat_vec)

        F, H, W, C = feature_maps.shape

        target_feat_vec = target_feat_vec.view(1, 1, 1, -1)  
        target_feat_vec = target_feat_vec.expand(F, H, W, C)
        
        # Create heatmaps using cosine-similarity
        norm_target_feat_vec = torch.norm(target_feat_vec, dim=-1, keepdim=True)
        norm_feat_vecs = torch.norm(feature_maps, dim=-1, keepdim=True)

        heatmaps = torch.sum(feature_maps * target_feat_vec, dim=-1, keepdim=True) 
        heatmaps = heatmaps / (norm_target_feat_vec * norm_feat_vecs)

        return heatmaps

    def __project_point_coordinates(self, point_input_space, spatial_input_space=256, spatial_latent_space=32):
        """
        Get Point Coordinates in Latent Space. Assumes quadratic spatial dimensions of input and latent space.

        Args:
            point_input_space: Tuple with coordinates of point. Format: (y, x, time)
            spatal_input_space: Height/Width of input space. Default: 256
            spatial_latent_space: Height/Width of latent space. Default: 256/8=32
        
        Returns:
            point_latent_space: Equivalent coordinates in latent space.
        """

        y, x, t = point_input_space

        factor = spatial_input_space / spatial_latent_space
        point_latent_space = (y/factor, x/factor, t)

        return point_latent_space
    
    def __get_feature_vec_bilinear(self, feature_maps, point):
        """
        Get feature vector at position described by coordinates of point by using bilinear sampling.

        Args:
            features_maps: Tensor with concatenation of feature maps. Dimension: [Frames, Height, Width, Channels]
            point: Tuple with coordinates of point to track (Point already projected to feature space). Format: (y, x, time)
        
        Returns: Tensor with feature vector at point location. Dimension: [Channels,]
        """

        feature_maps = torch.permute(feature_maps, (0, 3, 1, 2)) # F, C, H, W

        y, x, t = point

        feat_map_t = feature_maps[t].unsqueeze(0) # 1, C, H, W

        x_int = math.floor(x)
        x_frac = x - x_int

        y_int = math.floor(y)
        y_frac = y - y_int

        # Scale frac part to be in range [-1, 1]
        x_grid = ( (x_frac - 0) / (1 - 0) ) * (1 - -1) + -1
        y_grid = ( (y_frac - 0) / (1 - 0) ) * (1 - -1) + -1

        grid = torch.tensor(
            [[[[x_grid, y_grid]]]], dtype=torch.float32
        ).to(feature_maps.device)

        # 1xCx2x2 Tensor used for interpolation
        feat_map_t_2x2 = feat_map_t[:, :, y_int:y_int+2, x_int:x_int+2]

        feat_vec = torch.nn.functional.grid_sample(feat_map_t_2x2, grid, mode="bilinear", align_corners=True)

        return feat_vec

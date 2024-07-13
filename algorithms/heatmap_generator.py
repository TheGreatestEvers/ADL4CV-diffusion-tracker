import torch
import math
import torch.nn.functional as func

#device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class HeatmapGenerator():
    """
    Used for generating Heatmaps
    """

    def __init__(self) -> None:
        ...
    
    def generate(self, features, query_points):
        """
        Generates Heatmaps.

        Args:
            features: Tensor with concatenation of feature maps. Dimension: [Frames, Channels, Height, Width]
            query_points: Tensor of point coordinates to track. Dimensions: [N, 3] with point format: (Time, y, x)

        Returns:
            heatmaps: Tensor of Heatmaps. Dimension: [NumPoints, Frames, Height, Width]
        """
        
        F, C, H, W = features.shape
        N, _ = query_points.shape

        targets_feat_vecs = self.bilinear_sampler(features, query_points)

        heatmaps = torch.zeros(N, F, H, W, device=device)

        for i, target_feat_vec in enumerate(targets_feat_vecs):
            
            # Create heatmaps using cosine-similarity
            norm_target_feat_vec = torch.norm(target_feat_vec, dim=0, keepdim=True)
            norm_feat_vecs = torch.norm(features, dim=1, keepdim=True)

            heatmap = torch.sum(features * target_feat_vec.view(1, C, 1, 1), dim=1, keepdim=True) 
            heatmap = heatmap / torch.maximum(norm_target_feat_vec * norm_feat_vecs, 1e-8 * torch.ones_like(norm_feat_vecs))

            heatmaps[i] = heatmap.squeeze()

        return heatmaps

    def __project_point_coordinates(self, targets, spatial_input_space=256, spatial_latent_space=32):
        """
        Get Point Coordinates in Latent Space. Assumes quadratic spatial dimensions of input and latent space.

        Args:
            Tensor of point coordinates to track. Dimensions: [N, 3] with point format: (Time, y, x)
            spatal_input_space: Height/Width of input space. Default: 256
            spatial_latent_space: Height/Width of latent space. Default: 256/8=32
        
        Returns:
            point_latent_space: Equivalent coordinates in latent space.
        """

        factor = spatial_latent_space / spatial_input_space 
        targets[:, 1:] *= factor # Targets to latent space

        return targets
    
    def __get_feature_vec_bilinear(self, feature_maps, targets):
        """
        Get feature vector at position described by coordinates of point by using bilinear sampling.

        Args:
            features_maps: Tensor with concatenation of feature maps. Dimension: [Frames, Channels, Height, Width]
            targets: Tensor of point coordinates to track. Dimensions: [N, 3] with point format: (Time, y, x)
        
        Returns: Tensor with feature vectors at point locations. Dimension: [N, Channels]
        """

        N = targets.shape[0]
        feat_vecs = torch.zeros(N, feature_maps.shape[1], device=feature_maps.device)

        # Extract target coordinates
        t = targets[:, 0].long()  # Time dimension
        y = targets[:, 1]  # y dimension
        x = targets[:, 2]  # x dimension

        # Get integer and fractional parts
        x_int = x.floor().long()
        x_frac = x - x_int.float()
        y_int = y.floor().long()
        y_frac = y - y_int.float()

        # Gather values from tensor1 using advanced indexing
        Q11 = feature_maps[t, :, y_int, x_int]
        Q12 = feature_maps[t, :, y_int, x_int+1]
        Q21 = feature_maps[t, :, y_int+1, x_int]
        Q22 = feature_maps[t, :, y_int+1, x_int+1]

        # Interpolated values
        feat_vecs = Q11 * (1 - x_frac).view(N,1) * (1 - y_frac).view(N,1) \
            + Q12 * (x_frac).view(N,1) * (1 - y_frac).view(N,1) \
            + Q21 * (y_frac).view(N,1) * (1 - x_frac).view(N,1) \
            + Q22 * (x_frac).view(N,1) * (y_frac).view(N,1)

        return feat_vecs

    def bilinear_sampler(self, tensor, points, mode='bilinear'):
        """
        Args:
        - tensor: A tensor of shape (F, C, H, W)
        - points: A tensor of shape (N, 3) where each point has coordinates (t, y, x)
        - mode: Interpolation mode (default is 'bilinear')
        - mask: Boolean flag to return mask of valid points (default is False)

        Returns:
        - sampled_vectors: A tensor of shape (N, C) with sampled vectors
        - mask (optional): A tensor of shape (N, 1) indicating valid points if mask=True
        """
        F, C, H, W = tensor.shape
        N = points.shape[0]
        
        # Extract t, y, x coordinates
        t = points[:, 0].long()  # t should be integer indices for the first dimension
        y = points[:, 1]
        x = points[:, 2]
        
        # Normalize the y and x coordinates to [-1, 1]
        y = 2 * y / (H - 1) - 1
        x = 2 * x / (W - 1) - 1

        # Create the grid for grid_sample
        grids = torch.stack((x, y), dim=-1).view(N, 1, 1, 2)
        
        # Sample the tensor at each t index
        sampled_vectors = []

        for i in range(N):
            # Select the specific t entry
            img_slice = tensor[t[i]].unsqueeze(0)  # shape (1, C, H, W)
            # Sample using grid_sample
            sampled = func.grid_sample(img_slice, grids[i].unsqueeze(0), align_corners=True)
            sampled_vectors.append(sampled.squeeze())  # shape (C,)
            
        sampled_vectors = torch.stack(sampled_vectors)  # shape (N, C)

        return sampled_vectors

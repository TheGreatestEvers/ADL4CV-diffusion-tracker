import torch
import math

#device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class HeatmapGenerator:
    """
    Used for generating Heatmaps
    """

    def __init__(self) -> None:
        ...
    
    def generate(self, feature_maps, targets, mode="keep_feat_vec"):
        """
        Generates Heatmaps.

        Args:
            features_maps: Tensor with concatenation of feature maps. Dimension: [Frames, Channels, Height, Width]
            targets: Tensor of point coordinates to track. Dimensions: [N, 3] with point format: (Time, y, x)
            mode: Either "keep_feat_vec" or "resample_feat_vec". First uses feature vector from inital point to generate heatmaps for all
                  frames latter, latter resamples features vector after each point estimation 

        Returns:
            heatmaps: Tensor of Heatmaps. Dimension: [NumPoints, Frames, Height, Width]
        """
        feature_maps = feature_maps.to(device=device)
        targets = targets.to(device)

        if feature_maps.shape[-1] != feature_maps.shape[-2]:
            ValueError("Featuremaps should have dimensions [Frames, Channels, Height, Width]")
        
        F, C, H, W = feature_maps.shape
        N, _ = targets.shape
        
        if mode == "keep_feat_vec":

            targets_proj = self.__project_point_coordinates(targets, spatial_latent_space=H)

            targets_feat_vecs = self.__get_feature_vec_bilinear(feature_maps, targets_proj)


            heatmaps = torch.zeros(N, F, H, W, device=device, dtype=torch.float32)

            for i, target_feat_vec in enumerate(targets_feat_vecs):
                
                # Create heatmaps using cosine-similarity
                norm_target_feat_vec = torch.norm(target_feat_vec, dim=0, keepdim=True)
                norm_feat_vecs = torch.norm(feature_maps, dim=1, keepdim=True)

                heatmap = torch.sum(feature_maps * target_feat_vec.view(1, C, 1, 1), dim=1, keepdim=True) 
                heatmap = heatmap / torch.maximum(norm_target_feat_vec * norm_feat_vecs, 1e-8 * torch.ones_like(norm_feat_vecs))

                heatmaps[i] = heatmap.squeeze()

        elif mode == "resample_feat_vec": 

            NotImplementedError("Resample feat vec mode not implemented for multiple point tracking.")
            # tracker = ZeroShotTracker()
            # heatmaps = torch.zeros(F, 1, 32, 32)
            # for t in range(F):
            #     point_proj = self.__project_point_coordinates(target_coordinates)

            #     target_feat_vec = self.__get_feature_vec_bilinear(feature_maps, point_proj)

            #     feat_map = feature_maps[t]

            #     target_feat_vec = target_feat_vec.view(1, 1, -1)  
            #     target_feat_vec = target_feat_vec.expand(C, H, W)

            #     norm_target_feat_vec = torch.norm(target_feat_vec, dim=0, keepdim=True)
            #     norm_feat_vecs = torch.norm(feat_map, dim=0, keepdim=True)

            #     heatmap = torch.sum(feat_map * target_feat_vec, dim=0, keepdim=True) 
            #     heatmap = heatmap / (norm_target_feat_vec * norm_feat_vecs)

            #     heatmap = heatmap.unsqueeze(0)

            #     track_estimation = tracker.track(heatmap)[0]

            #     target_coordinates = (t+1, track_estimation[0].numpy(), track_estimation[1].numpy())

            #     heatmaps[t] = heatmap[0]

        else:
            raise ValueError("Mode has to be either keep_feat_vec or resample_feat_vec.")

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

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
                heatmap = heatmap / (norm_target_feat_vec * norm_feat_vecs)

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

        feat_vecs = torch.zeros(targets.shape[0], feature_maps.shape[1], dtype=torch.float64).to(feature_maps.device)

        for i, target in enumerate(targets):

            t, y, x = target
            t = t.item()
            y = y.item()
            x = x.item()

            feat_map_t = feature_maps[int(t)].unsqueeze(0) # 1, C, H, W

            x_int = math.floor(x)
            x_frac = x - x_int

            y_int = math.floor(y)
            y_frac = y - y_int

            # Scale frac part to be in range [-1, 1]
            x_grid = x_frac * 2 - 1
            y_grid = y_frac * 2 - 1

            grid = torch.tensor(
                [[[[x_grid, y_grid]]]], dtype=torch.float32
            ).to(feature_maps.device)

            # 1xCx2x2 Tensor used for interpolation
            feat_map_t_2x2 = feat_map_t[:, :, y_int:y_int+2, x_int:x_int+2]

            feat_vec = torch.nn.functional.grid_sample(feat_map_t_2x2, grid, mode="bilinear", align_corners=True).squeeze() # grid_sample works only with float32

            feat_vecs[i] = feat_vec

        return feat_vecs

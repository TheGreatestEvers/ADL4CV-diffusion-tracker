import torch
import numpy as np
from PIL import Image

class ZeroShotTracker:
    """
    Abstract trajectory of target point from precalculated heatmap
    """
    
    def __init__(self) -> None:
        pass

    def track(self, heatmaps, image_spatial_size=256):
        """
        Determines coordinates of point with highest value in heatmap.

        Args:
            heatmaps: Heatmaps with Dimensions: [Frames, Height_latent, Width_latent, 1]

        Returns:
            tracks: Tensor containing x and y coordinate of extimate in each frame. Dimensions: [Frames, 2]
        """

        # Scale heatmap from latent spatials to image spatials
        heatmaps = torch.permute(heatmaps, (0, 3, 1, 2)).to("cpu")
        #heatmaps = torch.nn.functional.interpolate(heatmaps, size=image_spatial_size, mode="bilinear", align_corners=True)

        F, _, H, W = heatmaps.shape

        # Get max coordinates for each frame 
        max_coordinates = torch.stack([(heatmaps[i,0]==torch.max(heatmaps[i,0])).nonzero()[0] for i in range(F)], dim=0).squeeze()
        
        # If only one frame is processed an extra dimension needs to be added
        if F == 1:
            max_coordinates = max_coordinates.unsqueeze(0)

        # Calculate estimation as weighted sum of points in the vicinity (max dist of 2 using L1) of max coordinate
        vicinity_distance = 2
        tracks = np.zeros((F, 2))

        for f in range(F):
            # Get vicinity coordinates
            vicinity_points = []
            y = max_coordinates[f, 0]
            x = max_coordinates[f, 1]
            for i in range(y - vicinity_distance, y + vicinity_distance + 1):
                for j in range(x - vicinity_distance, x + vicinity_distance + 1):
                    if 0 <= i < H and 0 <= j < W:  # Ensure the point is within the tensor boundaries
                        if abs(i - y) + abs(j - x) <= vicinity_distance:
                            vicinity_points.append((i, j))
            vicinity_points = np.array(vicinity_points)

            # Weighted sum of coordinates
            weighted_coordinates = np.zeros(2)
            normalization = 0
            for p in vicinity_points:
                weighted_coordinates = weighted_coordinates + p * heatmaps[f, 0, p[0], p[1]].numpy()
                normalization = normalization + heatmaps[f, 0, p[0], p[1]]
            weighted_coordinates = weighted_coordinates / normalization

            tracks[f, :] = weighted_coordinates
            
        tracks = tracks * 8

        return torch.from_numpy(tracks)


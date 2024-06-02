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
            heatmaps: Heatmaps with Dimensions: [Frames, NumPoints, 1, Height_latent, Width_latent]

        Returns:
            tracks: Tensor containing xy coordinate of each estimated point in each track. Dimensions: [NumPoints, Frames, 2]
        """

        # Scale heatmap from latent spatials to image spatials
        #heatmaps = torch.nn.functional.interpolate(heatmaps, size=image_spatial_size, mode="bilinear", align_corners=True)

        heatmaps = heatmaps.squeeze()
        F, N, H, W = heatmaps.shape
        
        scaling_factor = image_spatial_size / H

        tracks = torch.zeros(N, F, 2).to(heatmaps.device)
        for n in range(N):

            heatmaps_point = heatmaps[:, n, :, :]

            # Get max coordinates for each frame 
            max_coordinates = torch.stack([(heatmaps_point[i]==torch.max(heatmaps_point[i])).nonzero()[0] for i in range(F)], dim=0).squeeze()
            
            # If only one frame is processed an extra dimension needs to be added
            if F == 1:
                max_coordinates = max_coordinates.unsqueeze(0)

            # Calculate estimation as weighted sum of points in the vicinity (max dist of 2 using L1) of max coordinate
            vicinity_distance = 2
            track = torch.zeros((F, 2)).to(heatmaps.device)

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
                vicinity_points = torch.tensor(vicinity_points).to(heatmaps.device)

                # Weighted sum of coordinates
                weighted_coordinates = torch.zeros(2).to(heatmaps.device)
                normalization = 0
                for p in vicinity_points:
                    weighted_coordinates = weighted_coordinates + p * heatmaps[f, n, p[0], p[1]]
                    normalization = normalization + heatmaps[f, n, p[0], p[1]].item()
                weighted_coordinates = weighted_coordinates / normalization

                track[f, :] = weighted_coordinates
                
            track = track * (image_spatial_size / H) # Up to original coordinates
            tracks[n] = track

        return tracks


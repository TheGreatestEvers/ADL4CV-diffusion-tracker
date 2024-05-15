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
        heatmaps = torch.nn.functional.interpolate(heatmaps, size=image_spatial_size, mode="bilinear", align_corners=True)

        # Get max coordinates for each frame 
        max_coordinates = torch.stack([(heatmaps[i,0]==torch.max(heatmaps[i,0])).nonzero()[0] for i in range(heatmaps.size(0))], dim=0).squeeze()

        # Calculate estimation as weighted sum of points in the vicinity (max dist of 2 using L1) of max coordinate
        F, _, H, W = heatmaps.shape
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
            
            tracks = np.rint(tracks).astype("int") # Round track coordinates

        return torch.from_numpy(tracks)

    def place_marker_in_frames(self, frames, tracks, safe_as_gif=True):
        """
        Add red marker to frames at estimated point location.

        Args:
            frames: Video frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
            tracks: Estimated location of point in each frame (tensor). Dimensions: [Frames, 2]

        Returns:
            frames_marked: Marked frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
        """

        tracks = tracks.to("cpu").numpy()

        frames_marked = frames
        for i in range(tracks.shape[0]):
            y, x = tracks[i]
            frames_marked[i, y, x] = [255, 0, 0]
            frames_marked[i, y+1, x] = [255, 0, 0]
            frames_marked[i, y-1, x] = [255, 0, 0]
            frames_marked[i, y, x+1] = [255, 0, 0]
            frames_marked[i, y, x-1] = [255, 0, 0]
            frames_marked[i, y+1, x+1] = [255, 0, 0]
            frames_marked[i, y+1, x-1] = [255, 0, 0]
            frames_marked[i, y-1, x+1] = [255, 0, 0]
            frames_marked[i, y-1, x-1] = [255, 0, 0]
        

        if safe_as_gif:
            frames_gif = [Image.fromarray(f) for f in frames_marked]
            frames_gif[0].save("output/marked_frames.gif", save_all=True, append_images=frames_gif[1:], duration=100, loop=0)




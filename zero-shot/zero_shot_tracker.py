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
        heatmaps = torch.permute(heatmaps, (0, 3, 1, 2))
        heatmaps = torch.nn.functional.interpolate(heatmaps, size=image_spatial_size, mode="bilinear", align_corners=True)

        # Get max coordinates for each frame # DOES NOT DEAL WITH MULTIPLE MAXIMA !!!!
        tracks = torch.stack([(heatmaps[i,0]==torch.max(heatmaps[i,0])).nonzero()[0] for i in range(heatmaps.size(0))], dim=0).squeeze()

        return tracks

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




import torch
from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
from matplotlib import cm

def array_to_heatmap(data_slice):
    """
    Creates colorful heatmap from array with values between 0 and 1

        Args:
            array: Numpy array of Shape [H, W] with values between 0 and 1

        Returns:
            image: Image of colored heatmap
    """

    colormap = cm.get_cmap('viridis')

    # Apply colormap to normalized data
    heatmap_array = colormap(data_slice)

    # Convert the heatmap array (RGBA) to RGB
    heatmap_array = (heatmap_array[:, :, :3] * 255).astype(np.uint8)

    # Convert the RGB array to a PIL image
    heatmap_image = Image.fromarray(heatmap_array)

    return heatmap_image


def blend_images(heatmap_img, video_frame, alpha=0.35):
    """
    Blends the heatmap image with the video frame using the given alpha transparency.

    Args:
        heatmap_img: PIL Image of the heatmap
        video_frame: Numpy array of the video frame
        alpha: Transparency value for blending (0.0 to 1.0)

    Returns:
        blended_img: Blended PIL Image
    """
    video_img = Image.fromarray(video_frame)
    video_img = video_img.convert("RGBA")
    heatmap_img = heatmap_img.convert("RGBA")

    blended_img = Image.blend(video_img, heatmap_img, alpha)
    return blended_img


def safe_heatmap_as_gif(heatmaps, overlay_video=False, frames=None, scaled=True, spatial_input_space=256):
    """
    Safe heatmaps as colorful gif.

    Args:
        heatmaps: Heatmaps tensor with Dimension: [Frames, Height, Width, 1]
        overlay_video: Whether heatmap should be transparent and og video is shown on top
        frames: Numpy array of video frames, Dimensions: [F, H, W, 3]
        scaled: Determines whether to also safe scaled heatmaps
        spatial_input_space: Spatial size of image space
    """

    heatmaps = torch.permute(heatmaps, (0, 3, 1, 2))
    
    if scaled:
        heatmaps_scaled = torch.nn.functional.interpolate(heatmaps, size=spatial_input_space, mode="bilinear", align_corners=True)
        heatmaps_scaled = heatmaps_scaled.to("cpu").squeeze().numpy()
    
    heatmaps = heatmaps.to("cpu").squeeze().numpy()

    # def custom_transform(x):
    #     threshold = 0.7
    #     return np.where(x < threshold, x * 0.5, x)

    # # Apply custom transformation
    # heatmaps_scaled = custom_transform(heatmaps_scaled)

    # plt.hist(heatmaps_scaled[20], bins=30, edgecolor='black')
    # # Adding labels and title for better understanding
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Normally Distributed Data')
    # # Display the plot
    # plt.show()

    frames_gif = [array_to_heatmap(h) for h in heatmaps]
    imageio.mimsave('output/heatmaps.gif', frames_gif, fps=15, loop=0)
    #frames_gif[0].save("../output/heatmaps.gif", save_all=True, append_images=frames_gif[1:], duration=100, loop=0)

    if scaled:
        #frames_gif = [array_to_heatmap(h) for h in heatmaps_scaled]
        #imageio.mimsave('output/heatmaps_scaled.gif', frames_gif, fps=20)
        #frames_gif[0].save("../output/heatmaps_scaled.gif", save_all=True, append_images=frames_gif[1:], duration=100, loop=0)
        if overlay_video and frames is not None:
            frames_gif = [blend_images(array_to_heatmap(h), f) for h, f in zip(heatmaps_scaled, frames)]
        else:
            frames_gif = [array_to_heatmap(h) for h in heatmaps_scaled]
        imageio.mimsave('output/heatmaps_scaled.gif', frames_gif, fps=15, loop=0)


def place_marker_in_frames(frames, tracks, safe_as_gif=True, ground_truth_tracks=None):
        """
        Add red marker to frames at estimated point location.

        Args:
            frames: Video frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
            tracks: Estimated location of point in each frame (tensor). Dimensions: [Frames, 2]

        Returns:
            frames_marked: Marked frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
        """

        if torch.is_tensor(tracks):
            tracks = tracks.to("cpu").numpy()
        elif isinstance(tracks, np.ndarray):
            ...
        else:
            ValueError("Tracks has to be either tensor or numpy array.")

        # Round coordinates to int in case they are float for some reason
        tracks = np.rint(tracks).astype(int)

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

        if ground_truth_tracks:
            for i in range(ground_truth_tracks.shape[0]):
                y, x = ground_truth_tracks[i]
                frames_marked[i, y, x] = [0, 255, 0]
                frames_marked[i, y+1, x] = [0, 255, 0]
                frames_marked[i, y-1, x] = [0, 255, 0]
                frames_marked[i, y, x+1] = [0, 255, 0]
                frames_marked[i, y, x-1] = [0, 255, 0]
                frames_marked[i, y+1, x+1] = [0, 255, 0]
                frames_marked[i, y+1, x-1] = [0, 255, 0]
                frames_marked[i, y-1, x+1] = [0, 255, 0]
                frames_marked[i, y-1, x-1] = [0, 255, 0]
        

        if safe_as_gif:
            frames_gif = [Image.fromarray(f) for f in frames_marked]
            frames_gif[0].save("output/marked_frames.gif", save_all=True, append_images=frames_gif[1:], duration=200, loop=0)
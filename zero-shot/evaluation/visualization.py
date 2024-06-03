import os
import torch
from PIL import Image, ImageSequence, ImageDraw
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


def blend_images(heatmap_img, video_frame, alpha=0.6):
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


def safe_heatmap_as_gif(heatmaps, overlay_video=False, frames=None, scaled=True, spatial_input_space=256, folder_path='output'):
    """
    Safe heatmaps as colorful gif.

    Args:
        heatmaps: Heatmaps tensor with Dimension: [Frames, Height, Width, 1]
        overlay_video: Whether heatmap should be transparent and og video is shown on top
        frames: Numpy array of video frames, Dimensions: [F, H, W, 3]
        scaled: Determines whether to also safe scaled heatmaps
        spatial_input_space: Spatial size of image space
    """

    
    #heatmaps = torch.permute(heatmaps, (0, 3, 1, 2))
    heatmaps = heatmaps[:,0,:,:,:]
    
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
    imageio.mimsave(os.path.join(folder_path, 'heatmaps.gif'), frames_gif, fps=15, loop=0)
    #frames_gif[0].save("../output/heatmaps.gif", save_all=True, append_images=frames_gif[1:], duration=100, loop=0)

    if scaled:
        #frames_gif = [array_to_heatmap(h) for h in heatmaps_scaled]
        #imageio.mimsave('output/heatmaps_scaled.gif', frames_gif, fps=20)
        #frames_gif[0].save("../output/heatmaps_scaled.gif", save_all=True, append_images=frames_gif[1:], duration=100, loop=0)
        if overlay_video and frames is not None:
            frames_gif = [blend_images(array_to_heatmap(h), f) for h, f in zip(heatmaps_scaled, frames)]
        else:
            frames_gif = [array_to_heatmap(h) for h in heatmaps_scaled]
        imageio.mimsave(os.path.join(folder_path, 'scaled_heatmaps.gif'), frames_gif, fps=15, loop=0)


def place_marker_in_frames(frames, tracks, safe_as_gif=True, ground_truth_tracks=None, folder_path='output'):
        """
        Add red marker to frames at estimated point location.

        Args:
            frames: Video frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
            tracks: Estimated location of each point in each frame (tensor). Dimensions: [NumPoints, Frames, 2]
            safe_as_gif: Wether frames should be saved as gif
            ground_truth_tracks: Ground truth tracks (numpy array). If not None, green marker will be placed at GT location

        Returns:
            frames_marked: Marked frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
        """

        if torch.is_tensor(tracks):
            tracks = tracks.to("cpu").numpy()
        elif isinstance(tracks, np.ndarray):
            ...
        else:
            ValueError("Tracks has to be either tensor or numpy array.")
        
        N, F, _ = tracks.shape

        # Round coordinates to int in case they are float for some reason
        tracks = np.rint(tracks).astype(int)

        marked_frames = []
        for i, frame in enumerate(frames):
            frame_image = Image.fromarray(frame.astype('uint8'))
            draw = ImageDraw.Draw(frame_image)
            
            for n in range(N):
                # Get the coordinates for the marker from indices
                y, x = tracks[n, i]
                
                # Draw the marker as a filled circle
                draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 62, 150))

                if ground_truth_tracks is not None:
                    if N != 1:
                        ValueError("Show ground truth only possible when only one point is tracked.")
                    y_gt, x_gt = ground_truth_tracks[i]
                    draw.ellipse((x_gt-2, y_gt-2, x_gt+2, y_gt+2), fill=(0, 255, 0))

                
            # Append the modified frame to the list
            marked_frames.append(np.array(frame_image))

        if safe_as_gif:
            imageio.mimsave(os.path.join(folder_path, 'marked_frames.gif'), marked_frames, fps=15, loop=0)
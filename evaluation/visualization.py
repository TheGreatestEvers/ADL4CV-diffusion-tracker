import os
import torch
from PIL import Image, ImageSequence, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
from matplotlib import cm
from sklearn.manifold import TSNE
import seaborn as sns
import cv2

def do_tsne(features, perplexity=7):
    N, F, C = features.shape

    features = features.reshape(-1, C) #Shape (N*F, C)

    features = features.detach().numpy()

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features)
    features_tsne = features_tsne.reshape(N, F, 2) #Shape (N, F, 3)

    return features_tsne

def gather_features(features, tracks):
    F, C, H, W = features.shape
    N, F, _ = tracks.shape

    h_indices = (tracks[..., 0] * H / 256).astype(np.int64)
    w_indices = (tracks[..., 1] * W / 256).astype(np.int64)
    
    gathered_features = torch.zeros(N, F, C, dtype=features.dtype, device=features.device)
    
    for n in range(N):
        for f in range(F):
            h_idx = h_indices[n, f]
            w_idx = w_indices[n, f]
            gathered_features[n, f] = features[f, :, h_idx, w_idx]

    return gathered_features


def visualize_tsne(features, tracks, image, query_points, refined_features=None, folder_path='plots'):
    N, F, _ = tracks.shape

    gathered_features_1 = gather_features(features, tracks)
    feature_tsne_1 = do_tsne(gathered_features_1)
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 14})

    if refined_features is None:
        ax1 = fig.add_subplot(121)
        ax1.imshow(image)
        
        ax1.set_title('Query Points')
        ax1.axis('off')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title('t-SNE feature similarity')

        colors = sns.color_palette("hsv", N)

        for i in range(N):
            ax1.scatter(query_points[i, 2], query_points[i, 1], color=colors[i], s=30)
            ax2.scatter(feature_tsne_1[i, :, 0], feature_tsne_1[i, :, 1], feature_tsne_1[i, :, 2], color=colors[i])
    
    else: 
        gathered_features_2 = gather_features(refined_features, tracks)
        feature_tsne_2 = do_tsne(gathered_features_2)

        ax1 = fig.add_subplot(131)
        ax1.imshow(image)
        
        ax1.set_title('Query Points')
        ax1.axis('off')

        ax2 = fig.add_subplot(132)#, projection='3d')
        ax2.set_title('t-SNE Raw Features')

        ax3 = fig.add_subplot(133)#, projection='3d')
        ax3.set_title('t-SNE Refined Features')

        colors = sns.color_palette("hsv", N)

        for i in range(N):
            ax1.scatter(query_points[i, 2], query_points[i, 1], color=colors[i], s=30)
            #ax2.scatter(feature_tsne_1[i, :, 0], feature_tsne_1[i, :, 1], feature_tsne_1[i, :, 2], color=colors[i])
            #ax3.scatter(feature_tsne_2[i, :, 0], feature_tsne_2[i, :, 1], feature_tsne_2[i, :, 2], color=colors[i])
            ax2.scatter(feature_tsne_1[i, :, 0], feature_tsne_1[i, :, 1], color=colors[i])
            ax3.scatter(feature_tsne_2[i, :, 0], feature_tsne_2[i, :, 1], color=colors[i])

    os.makedirs(folder_path, exist_ok=True)
    figure_path = os.path.join(folder_path, 'tsne_plot.svg')

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.show()


def visualize_multiple_heatmaps(video, query_point, heatmaps, time_indices, folder_path='plots'):
    num_plots = len(time_indices) + 1
    PLT_HEIGHT = 5

    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(num_plots*PLT_HEIGHT,PLT_HEIGHT + 1))

    t, y, x = query_point.astype(np.int64)

    ax1 = fig.add_subplot(100 + num_plots * 10 + 1)
    ax1.imshow(video[t])  
    ax1.set_title('Query Point')
    ax1.axis('off')
    ax1.scatter(x, y, color='b', s=50)

    for i, time_idx in enumerate(time_indices):
        heatmap = heatmaps[time_idx]
        image = video[time_idx]

        ax = fig.add_subplot(100 + num_plots * 10 + i + 2)
        ax.set_title('t = ' + str(time_idx))
        ax.axis('off')
        ax.imshow(image)
        sns.heatmap(heatmap, ax=ax, alpha=0.5, cmap='viridis', cbar=False)

    os.makedirs(folder_path, exist_ok=True)
    figure_path = os.path.join(folder_path, 'multiple_heatmaps.png')

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.show()

def visualize_pred_error(video, query_points, pred_points, target_points, occluded, time_indices, folder_path='plots'):
    N, _ = query_points.shape

    num_plots = len(time_indices) + 1
    PLT_HEIGHT = 5

    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(num_plots*PLT_HEIGHT,PLT_HEIGHT + 1))

    colors = sns.color_palette("hsv", N)

    t = query_points[0, 0].astype(np.int64)

    ax1 = fig.add_subplot(100 + num_plots * 10 + 1)
    ax1.imshow(video[t])  
    ax1.set_title('Query Point')
    ax1.axis('off')

    for i in range(N):
        ax1.scatter(query_points[i, 2], query_points[i, 1], color=colors[i], s=30)

    for i, time_idx in enumerate(time_indices):
        image = video[time_idx]

        ax = fig.add_subplot(100 + num_plots * 10 + i + 2)
        ax.set_title('t = ' + str(time_idx))
        ax.axis('off')
        ax.imshow(image)

        for idx in range(N):
            if occluded[idx, time_idx]:
                continue

            y = (pred_points[idx, time_idx, 0], target_points[idx, time_idx, 0])
            x = (pred_points[idx, time_idx, 1], target_points[idx, time_idx, 1])

            ax.plot(x, y, color=colors[idx], marker='o')

    os.makedirs(folder_path, exist_ok=True)
    figure_path = os.path.join(folder_path, 'pred_error.png')

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.show()

def visualize_heatmaps(video, heatmaps, pred_points, target_points):
    """
    Visualize all N heatmaps from a PyTorch tensor of shape (N, H, W).
    
    Args:
        heatmaps (torch.Tensor): Tensor of shape (N, H, W) containing the heatmaps.
    """
    N, H, W = heatmaps.shape
    # Determine the number of rows and columns for the subplots
    cols = int(torch.ceil(torch.sqrt(torch.tensor(N)).float()))
    rows = int(torch.ceil(torch.tensor(N).float() / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), dpi=50)
    
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < N:
            #ax.imshow(video[i].permute(1, 2, 0).detach().cpu().numpy())
            ax.imshow(heatmaps[i].detach().cpu().numpy(), cmap='viridis')
            ax.set_title(f'Heatmap {i+1}')
            x, y = pred_points[i].detach().cpu().numpy()
            ax.plot(y, x, 'bo')  # Plot the point in blue with a circle marker
            x, y = target_points[i].detach().cpu().numpy()
            ax.plot(y, x, 'ro')  # Plot the point in blue with a circle marker
        ax.axis('off')
    
    plt.tight_layout()
    #plt.show()

    return plt


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


def safe_heatmap_as_gif(heatmaps, overlay_video=False, frames=None, folder_path='plots'):
    """
    Safe heatmaps as colorful gif.

    Args:
        heatmaps: Heatmaps tensor with Dimension: [Frames, Height, Width]
        overlay_video: Whether heatmap should be transparent and og video is shown on top
        frames: Numpy array of video frames, Dimensions: [F, H, W, 3]
        scaled: Determines whether to also safe scaled heatmaps
        spatial_input_space: Spatial size of image space
    """

    if overlay_video and frames is not None:
        frames_gif = [blend_images(array_to_heatmap(h), f, alpha=0.4) for h, f in zip(heatmaps, frames)]
    else:
        frames_gif = [array_to_heatmap(h) for h in heatmaps]

    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, 'heatmap.gif')
    imageio.mimsave(file_path, frames_gif, fps=15, loop=0)



def save_pred_points_as_gif(video, pred_points, target_points, occluded, point_indices, folder_path='plots'):
    """
    Add red marker to frames at estimated point location and save as a GIF.

    Args:
        frames: Video frames (numpy array) with Dimensions: [Frames, Height, Width, 3]
        pred_points: Estimated location of each point in each frame (tensor). Dimensions: [NumPoints, Frames, 2]
        target_points: Ground truth tracks (numpy array) with Dimensions: [NumPoints, Frames, 2], if target_points = None target points are not displayed
        occluded: pred_points and target_points are not displayed if occluded = True [NumPoints, Frames]
        point_indices: list of indices in NumPoints indicating which points to plot
        gif_path: Path to save the output GIF
    """
    
    F, H, W, _ = video.shape
    N, F, _ = pred_points.shape
    
    frames_with_markers = []
    
    for frame_idx in range(F):
        frame = video[frame_idx].copy()
        
        for point_idx in point_indices:
            if point_idx < N and not occluded[point_idx, frame_idx]:
                # Draw predicted points
                pred_y, pred_x  = pred_points[point_idx, frame_idx]
                cv2.circle(frame, (int(pred_x), int(pred_y)), 5, (0, 0, 255), -1)  # Red color for predicted points

                if target_points is not None:
                    # Draw target points
                    target_y, target_x = target_points[point_idx, frame_idx]
                    cv2.circle(frame, (int(target_x), int(target_y)), 5, (255, 0, 0), -1)  # Blue color for target points
        frames_with_markers.append(frame)
    
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, 'pred_points.gif')
    imageio.mimsave(file_path, frames_with_markers, fps=10)
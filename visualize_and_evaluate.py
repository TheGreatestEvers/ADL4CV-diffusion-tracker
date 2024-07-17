import os
from evaluation.visualization import visualize_tsne, visualize_multiple_heatmaps, visualize_pred_error, safe_heatmap_as_gif, save_pred_points_as_gif
from algorithms.feature_extraction_loading import FeatureDataset, concatenate_video_features
from learning_based.residual_block import ResidualFeatureBlock
from algorithms.heatmap_generator import HeatmapGenerator
from algorithms.heatmap_processor import HeatmapProcessor
from evaluation.evaluation_datasets import compute_tapvid_metrics
import torch

VIDEO_IDX = 29
BATCH_SIZE = 8
N_QUERY_POINTS = 5
N_FRAMES = 32
HEATMAP_QUERY_POINT = 2
MODEL_NAME = 'glowing-puddle-375.pt'


if __name__ == '__main__':
    
    dataset = FeatureDataset(feature_dataset_path='/home/max/ADL4CV/features/davis')
    data = dataset[VIDEO_IDX]

    features = data['better_feature'][:N_FRAMES]
    features_raw = data['features']
    features_raw = concatenate_video_features({'up_block': features_raw['up_block'], 'down_block': features_raw['down_block']})[:N_FRAMES]

    folder_path = os.path.join('plots', 'video_' + str(VIDEO_IDX) + '_frames_' + str(N_FRAMES))

    query_points = data["query_points"][0]
    target_points = data["target_points"][..., [1, 0]][0][:, :N_FRAMES]
    occluded = data['occluded'][0, :, :N_FRAMES]
    target_points = target_points[query_points[:, 0] < N_FRAMES]
    occluded = occluded[query_points[:, 0] < N_FRAMES]
    query_points = query_points[query_points[:, 0] < N_FRAMES]

    video = data['video'][0][:N_FRAMES]
    query_image = video[0]

    red_target_points = target_points[query_points[:, 0] == 0][:N_QUERY_POINTS]
    red_occluded = occluded[query_points[:, 0] == 0][:N_QUERY_POINTS]
    red_query_points = query_points[query_points[:, 0] == 0][:N_QUERY_POINTS]

    pretrained_path = os.path.join('models', MODEL_NAME)
    checkpoint = torch.load(pretrained_path)
    residual_block = ResidualFeatureBlock(intermediate_channels=[8, 16, 32, 64], n_output_channels=64)
    residual_block.load_state_dict(checkpoint['residual_state_dict'])
    heatmap_generator = HeatmapGenerator()
    heatmap_processor = HeatmapProcessor()
    heatmap_processor.load_state_dict(checkpoint['processor_state_dict'])
    heatmap_processor = heatmap_processor.to(device='cuda')

    residual_block.eval()
    heatmap_processor.eval()

    pred_points = []
    heatmaps = []
    
    with torch.no_grad():
        refined_features = features + residual_block(torch.tensor(video).permute(0, 3, 1, 2).float() / 255.)
        for i in range(0, query_points.shape[0], BATCH_SIZE):
            heatmap = heatmap_generator.generate(refined_features, torch.tensor(query_points[i:min(query_points.shape[0], i+BATCH_SIZE)]).float())
            heatmaps.append(heatmap)
            pred_points.append(heatmap_processor.predictions_from_heatmap(heatmap))

        pred_points = torch.cat(pred_points, dim=0)
        heatmaps = torch.cat(heatmaps, dim=0)

        heatmaps = heatmaps.cpu().numpy()
        pred_points = pred_points.cpu().numpy()

        red_pred_points = pred_points[query_points[:, 0] == 0][:N_QUERY_POINTS]

    visualize_tsne(features_raw, red_target_points, query_image, red_query_points, refined_features, folder_path=folder_path)

    visualize_multiple_heatmaps(video, red_query_points[HEATMAP_QUERY_POINT], heatmaps[HEATMAP_QUERY_POINT], time_indices=[10, 20, 30], folder_path=folder_path)

    visualize_pred_error(video, red_query_points, red_pred_points, red_target_points, red_occluded, time_indices=[10, 20, 30], folder_path=folder_path)

    safe_heatmap_as_gif(heatmaps[HEATMAP_QUERY_POINT], overlay_video=True, frames=video, folder_path=folder_path)

    save_pred_points_as_gif(video, red_pred_points, target_points=red_target_points, occluded=red_occluded, point_indices=[0], folder_path=folder_path)

    metrics = compute_tapvid_metrics(query_points=query_points[None], gt_occluded=occluded[None], gt_tracks=target_points[None], pred_occluded=occluded[None], pred_tracks=pred_points[None], query_mode='strided')

    print(metrics)

    file_path = os.path.join(folder_path, 'metrics.txt')
    with open(file_path, 'w') as metrics_file:
        metrics_file.write(str(metrics))


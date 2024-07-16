import os
import yaml
import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from algorithms.feature_extraction_loading import FeatureDataset
from algorithms.utils import read_config_file, feature_collate_fn
from evaluation.evaluation_datasets import compute_tapvid_metrics
from learning_based.learn_upsample_tracker import LearnUpsampleTracker
import evaluation.visualization as viz
from math import ceil

device = "cuda"

def get_query_batch(query_points, target_points, occluded, trackgroup, batch_size=8, shuffle=True, drop_last=False):
    """
    Yields a tuple containing one batch of query_points, target_points, occluded, trackgroup
    """
    num_points = query_points.shape[0]

    if shuffle:
        permutation = np.random.permutation(num_points)
        query_points = query_points[permutation]
        target_points = target_points[permutation]
        occluded = occluded[permutation]
        trackgroup = trackgroup[permutation]

    if drop_last:
        num_batches = num_points // batch_size
    else:
        num_batches = ceil(num_points / batch_size)
    
    for i in range(num_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, num_points)

        yield (
            query_points[start:end],
            target_points[start:end],
            occluded[start:end],
            trackgroup[start:end],
        )

def main():
    dataset = FeatureDataset(feature_dataset_path="features/davis/")
    dataloader = DataLoader(dataset, 1, shuffle=True, collate_fn=feature_collate_fn)

    model = LearnUpsampleTracker(next(iter(dataloader))[0]["features"])
    model.load_state_dict(torch.load('trained_upsample_1.pth'))
    model.to(device)
    #model.eval()

    for i, data in enumerate(dataloader):
        data = data[0]

        feature_dict = data['features']
        for block_name, block_feat_list in feature_dict.items():
            for i in range(len(block_feat_list)):
                feature_dict[block_name][i] = feature_dict[block_name][i].to(dtype=torch.float32).to(device)
        
        all_points = []
        all_occluded = []

        for query_batch in get_query_batch(data["query_points"][0], data["target_points"][0], \
            data["occluded"][0], data["trackgroup"][0]):

            query_points, target_points, occluded, trackgroup = query_batch

            query_points = torch.tensor(query_points, dtype=torch.float32, device=device)
            target_points = torch.tensor(target_points[..., [1, 0]], dtype=torch.float32, device=device)
            occluded = torch.tensor(occluded, dtype=torch.float32, device=device)
            trackgroup = torch.tensor(trackgroup, dtype=torch.float32, device=device)

            #with autocast():
            #    pred_points, _ = self.model(feature_dict, query_points)

            with torch.no_grad():
                pred_points, pred_occlusions = model(feature_dict, query_points)
            #pred_points = pred_points * (1 - occluded).unsqueeze(-1)

            all_points.append(pred_points)
            all_occluded.append(occluded)


        all_points = torch.cat(all_points)
        all_occluded = torch.cat(all_occluded)

        viz.save_pred_points_as_gif(data["video"][0], all_points, all_occluded, folder_path=".")

        # Step 1: Swap the coordinates (y,x) -> (x,y)
        all_points = all_points[..., [1, 0]]
        # Step 2: Convert to numpy
        all_points = all_points.cpu().numpy()
        # Step 3: Add an extra dimension
        all_points = np.expand_dims(all_points, axis=0)
        metrics = compute_tapvid_metrics(query_points=data["query_points"], gt_occluded=data["occluded"], \
            gt_tracks=data["target_points"], pred_occluded=data["occluded"], pred_tracks=all_points, query_mode='strided')
        
        print(metrics)

        import json

        # Saving to a JSON file
        with open('data.json', 'w') as file:
            json.dump(metrics, file, indent=4)


        

if __name__ == "__main__":
    main()
import os
import yaml
#import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from algorithms.feature_extraction_loading import FeatureDataset
from algorithms.utils import read_config_file, feature_collate_fn
from evaluation.evaluation_datasets import compute_tapvid_metrics
from torch.utils.tensorboard import SummaryWriter
from learning_based.weighted_features_tracker import WeightedFeaturesTracker, WeightedHeatmapsTracker

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ZeViPo():
    def __init__(self, config_file: str):

        self.config = read_config_file(config_file)

        self.dataset = FeatureDataset(feature_dataset_path=self.config['dataset_dir'])
        self.dataloader = DataLoader(self.dataset, self.config['batch_size'], shuffle=True, collate_fn=feature_collate_fn)

        self.model = WeightedFeaturesTracker(next(iter(self.dataloader))[0]["features"])

        self.loss_fn = torch.nn.HuberLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        self.epochs = self.config['epochs']

    def train_one_epoch(self, epoch_index, logger):
        for i, data in enumerate(self.dataloader):
            data = data[0]

            indices = torch.arange(0, 10, 1)

            feature_dict = data['features']
            query_points = torch.cat([torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0) for p in data['query_points'][0]], dim=0)[indices]
            gt_points = torch.tensor(data['target_points'][0][..., [1, 0]], dtype=torch.float32, device=device)[indices]
            visible = 1 - torch.tensor(data['occluded'][0], dtype=torch.float32, device=device)[indices]

            self.optimizer.zero_grad()
            pred_points = self.model(feature_dict, query_points)
            
            losses = self.loss_fn(gt_points, pred_points)
            loss = torch.mean(losses * visible.unsqueeze(-1))

            loss.backward()
            self.optimizer.step()

            print(loss.item())

            iteration = epoch_index * len(self.dataloader) + i + 1
            logger.add_scalar('Loss/train', loss.item(), iteration)


    def eval_random(self):
        tracking_accuracy = []

        for i, data in enumerate(self.dataloader):
            batch_query_point = []
            batch_gt_occluded = []
            batch_gt_point = []
            batch_pred_point = []

            data = data[0]

            rnd_idx = torch.randint(0, len(data['query_points'][0]))

            feature_dict = data['features']
            query_point = torch.tensor(data['query_points'][0][rnd_idx], device=device).unsqueeze(0)
            
            pred_point = self.model(feature_dict, query_point)

            batch_pred_point.append(pred_point.cpu().numpy())
            batch_query_point.append(query_point.cpu().numpy())
            batch_gt_point.append(data['target_points'][0][rnd_idx][None])
            batch_gt_occluded.append(data['occluded'][0])

            metrics = compute_tapvid_metrics(query_points=np.array(batch_query_point), gt_occluded=np.array(batch_gt_occluded), gt_tracks=np.array(batch_gt_point), pred_occluded=np.array(batch_gt_occluded), pred_tracks=np.array(batch_pred_point), query_mode='strided')
            tracking_accuracy.append(metrics['average_pts_within_thresh'])

        return np.mean(tracking_accuracy)



    def train(self):

        logger = SummaryWriter(log_dir=self.config['log_dir'])
        # wandb.init(entity=self.config['wandb']['entity'],
        #            project=self.config['wandb']['project'],
        #            sync_tensorboard=True,
        #            config=self.config)
        
        for epoch in tqdm(range(self.epochs)):

            self.model.train(True)
            self.train_one_epoch(epoch, logger)

            self.model.eval()
            mean_tracking_accuracy = self.eval_random()

            logger.add_scalar('Acc/train', mean_tracking_accuracy, epoch)

            logger.flush()
        
        #wandb.finish()


if __name__ == '__main__':
    tracker = ZeViPo('configs/config.yml')
    tracker.train()
    
    print(tracker.eval_random())
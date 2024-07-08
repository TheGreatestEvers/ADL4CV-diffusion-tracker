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
from learning_based.model import FeatureDictProcessor, TrackingModel
from learning_based.residual_block import ResidualFeatureBlock
import time
import torch.nn.functional as func

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def extract_bilinearly(tensor, loc):
    # Tensor has shape CxHxW

    y = loc[0]  # y dimension
    x = loc[1]  # x dimension

    # Get integer and fractional parts
    x_int = x.floor().long()
    x_frac = x - x_int.float()
    y_int = y.floor().long()
    y_frac = y - y_int.float()

    # Gather values from tensor1 using advanced indexing
    Q11 = tensor[:, y_int, x_int]
    Q12 = tensor[:, y_int, x_int+1]
    Q21 = tensor[:, y_int+1, x_int]
    Q22 = tensor[:, y_int+1, x_int+1]

    # Interpolated values
    sampled = Q11 * (1 - x_frac) * (1 - y_frac) \
        + Q12 * (x_frac) * (1 - y_frac) \
        + Q21 * (y_frac) * (1 - x_frac) \
        + Q22 * (x_frac) * (y_frac)

    return sampled

class SelfsupervisedDiffusionTracker():

    def __init__(self) -> None:

        self.config = read_config_file("configs/config.yaml")

        dataset = FeatureDataset(feature_dataset_path=self.config['dataset_dir'])
        self.data = dataset[0]
        feature = torch.load("swan_feature_tensor.pt").to(device) # fix

        F,C,H,W = feature.shape
        feature = feature.permute((0, 2, 3, 1)).contiguous().view(F*H,W,C)
        feature = func.interpolate(feature, size=64)
        feature = feature.view(F,H,W,64).permute(0,3,1,2)

        self.data["features"] = feature

        self.track_model = TrackingModel().to(device)
        self.residual_block = ResidualFeatureBlock(intermediate_channels=[32], n_output_channels=64).to(device)

        params = list(self.track_model.parameters()) + list(self.residual_block.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config['learning_rate'])        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.99)

        self.mse = torch.nn.MSELoss()
        self.huber = torch.nn.HuberLoss()

    def loss_long(self, og_query_points, pred_endpoints, reverse_feat_dict):
        """
        Use predicted point at end of sequence as new query point. Goal is to predict original query point. Stepwise
        over all frames.
        """

        N = pred_endpoints.shape[0]

        prepended_zeros = torch.zeros(N, 1).to(device)
        new_queries = torch.cat((prepended_zeros, pred_endpoints[:, ]), dim=1)
        pred = self.track_model.forward_stepwise(reverse_feat_dict, new_queries)
        loss_long = self.mse(og_query_points[:, 1:], pred[:, -1])

        return loss_long
    
    def loss_skip(self, og_query_points, pred_endpoints, reverse_slim_feat_dict):
        """
        Use predicted point at end of sequence as new query point. Goal is to predict original query point.

        Slim feature dict is in reversed frame order, so it contains the last sequence frame and then the first sequence frame.
        """

        N = pred_endpoints.shape[0]

        prepended_zeros = torch.zeros(N, 1).to(device)
        new_queries = torch.cat((prepended_zeros, pred_endpoints), dim=1)
        pred = self.track_model.forward_skip(reverse_slim_feat_dict, new_queries)
        loss_skip = self.mse(og_query_points[:, 1:], pred[:, -1])

        return loss_skip

    def loss_feature_comparison(self, features, loc1, loc2):

        f1 = extract_bilinearly(features[0], loc1)
        f2 = extract_bilinearly(features[-1], loc2)

        return -torch.dot(f1, f2)

    def train(self):

        wandb.init(entity=self.config['wandb']['entity'],
          project=self.config['wandb']['project'],
          #mode="disabled",
          config=self.config)
        
        
        features = self.data["features"].to(device)
        target_points = torch.tensor(self.data["target_points"][..., [1, 0]], dtype=torch.float32, device=device)
        occluded = torch.tensor(self.data["occluded"], dtype=torch.float32, device=device)
        query_points = torch.tensor(self.data["query_points"], dtype=torch.float32, device=device)[0]
        query_points = query_points[query_points[:, 0] == 0]
        
        video = self.data["video"][0] / 255.0
        video_tensor = torch.tensor(video, device=device, dtype=torch.float32).permute(0, 3, 1, 2)

        features = features[:16]
        video_tensor = video_tensor[:16]

        total_iterations = self.config["total_iterations"]

        for iter in tqdm(range(total_iterations), desc="Train iteration"):

            ## Train
            self.track_model.train()

            self.optimizer.zero_grad()

            running_loss = torch.zeros(1).to(device)


            refined_features = features + self.residual_block(video_tensor)
            reversed_refined_features = torch.flip(refined_features, dims=[0])

            pred_points_stepwise = self.track_model.forward_stepwise(refined_features, query_points)
            pred_points_skip = self.track_model.forward_skip(refined_features, query_points)

            for i in range(2, refined_features.shape[0]):

                print(i-2)

                loss_long = self.loss_long(query_points, pred_points_stepwise[:,i-1], reversed_refined_features[-i:])
                loss_skip = self.loss_skip(query_points, pred_points_skip[:, i-1], reversed_refined_features[(-i, -1), ...])

                running_loss += loss_long + loss_skip

            print(np.array(torch.cuda.mem_get_info()) / 1e9)

            running_loss.backward()
            self.optimizer.step()
            
            wandb.log({"running_loss": running_loss})
        

            # ## Eval
            # if iter % 50 == 0 and iter > 1:
            #     self.feature_processor.eval()
            #     self.track_model.eval()

            #     features = self.feature_processor(feature_dict)
            #     pred_points = self.track_model(features, query_points)

            #     pred_points = pred_points * (1 - self.occluded).unsqueeze(-1)
            #     eval_loss = self.mse(target_points, pred_points)

            #     wandb.log({"loss": eval_loss})

        wandb.finish()    


if __name__ == "__main__":
    dt = SelfsupervisedDiffusionTracker()
    dt.train()
    torch.save(dt.model.state_dict(), 'unsupervised.pth')









            






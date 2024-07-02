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

        self.dataset = FeatureDataset(feature_dataset_path=self.config['dataset_dir'])

        self.feature_processor = FeatureDictProcessor(self.dataset[0]["features"])
        self.track_model = TrackingModel()

        params = [self.feature_processor.parameters(), self.track_model.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=self.config['learning_rate'])        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.99)

        self.mse = torch.nn.MSELoss()
        self.huber = torch.nn.HuberLoss()

    def preprocess_video(self, video):
        """
        Preprocesses frames of video to be applicable to RAFT optical flow model.
        Video is numpy array
        """

        video = video / 255.0
        frames = frames.transpose(0, 3, 1, 2)
        video_tensor = torch.tensor(frames, dtype=torch.float32)

        return video_tensor

    def sequence_loader(self, feature_dict):
        """
        Yield feat dict for sequences of frames. Sequence lengths: 2, 3, ..., F
        """

        F = feature_dict["up_block"][1].shape[0]

        for seq_end in range(1, F):
            seq_feat_dict = {key: [tensor[:seq_end] for tensor in value] for key, value in feature_dict.items()}

            yield seq_feat_dict

    def loss_long(self, og_query_point, pred_endpoint, reverse_feat_dict):
        """
        Use predicted point at end of sequence as new query point. Goal is to predict original query point. Stepwise
        over all frames.
        """

        pred = self.model(reverse_feat_dict, pred_endpoint)
        loss_long = self.mse(og_query_point, pred)

        # Maybe also add loss comparison of both feature vectors

        return loss_long
    
    def loss_skip(self, og_query_point, pred_endpoint, reverse_slim_feat_dict):
        """
        Use predicted point at end of sequence as new query point. Goal is to predict original query point.

        Slim feature dict is in reversed frame order, so it contains the last sequence frame and then the first sequence frame.
        """

        pred = self.model(reverse_slim_feat_dict, pred_endpoint)
        loss_skip = self.mse(og_query_point, pred)

        # Maybe also add loss comparison of both feature vectors

        return loss_skip

    def loss_feature_comparison(self, features, loc1, loc2):

        f1 = extract_bilinearly(features[0], loc1)
        f2 = extract_bilinearly(features[-1], loc2)

        return -torch.dot(f1, f2)

    def loss_of(self, video, pred_points):
        """
        Optical Flow loss.
        """

        running_loss = torch.zero(1)
        for i in range(video.shape[0] - 1):
            frame1 = video[i]
            frame2 = video[i + 1]

            flow_low, flow_up = self.of_raft(frame1, frame2, iters=20, test_mode=True)

            point1 = pred_points[i]

            # Obtain point2 prediction by optical flow and compare with actual point prediction
            flow_vector = extract_bilinearly(flow_up[0])
            point2_of = point1 + flow_vector

            point2_tracker = pred_points[i + 1]

            running_loss += self.huber(point2_of, point2_tracker)

        return running_loss

    def train(self):
        wandb.init(entity=self.config['wandb']['entity'],
          project=self.config['wandb']['project'],
          config=self.config)
        
        data = self.dataset[0]
        
        feature_dict = data["features"]
        target_points = torch.tensor(data["target_points"][..., [1, 0]], dtype=torch.float32, device=device)
        occluded = torch.tensor(occluded, dtype=torch.float32, device=device)
        query_points = torch.tensor(data["query_points"], dtype=torch.float32, device=device)
        
        video = self.preprocess_video(data["video"])

        total_iterations = self.config["total_iterations"]

        for iter in tqdm(range(total_iterations)):

            ## Train
            self.feature_processor.train()
            self.track_model.train()

            loss_long_running = torch.zeros(1)
            loss_skip_running = torch.zeros(1)
            loss_feature_comparison_running = torch.zeros(1)

            self.optimizer.zero_grad()

            for sequence_feat_dict in enumerate(self.sequence_loader(feature_dict)):

                features = self.feature_processor(sequence_feat_dict)

                pred_points = self.track_model(features, query_points) # NxFx2

                reversed_features = torch.flip(features, dims=0)

                loss_long = self.loss_long(query_points, pred_points[:, -1], reversed_features)
                loss_skip = self.loss_skip(query_points, pred_points[:, -1], reversed_features[(0, -1), ...])
                loss_feature_comparison = self.loss_feature_comparison(features, query_points, pred_points[:, -1])

                loss_long_running += loss_long
                loss_skip_running += loss_skip
                loss_feature_comparison_running += loss_feature_comparison
            
            # Optical flow loss
            loss_of = self.loss_of(video, pred_points)

            loss = 1/4 * loss_long_running \
                 + 1/4 * loss_skip_running \
                 + 1/4 * loss_feature_comparison_running \
                 + 1/4 * loss_of
            
            wandb.log({"loss_long": loss_long})
            wandb.log({"loss_skip": loss_skip})
            wandb.log({"loss_feature_comparison": loss_feature_comparison})
            wandb.log({"loss_of": loss_of})
            wandb.log({"loss": loss})
            
            loss.backward()
            self.optimizer.step()

            ## Eval
            if iter % 50 == 0 and iter > 1:
                self.feature_processor.eval()
                self.track_model.eval()

                features = self.feature_processor(feature_dict)
                pred_points = self.track_model(features, query_points)

                pred_points = pred_points * (1 - self.occluded).unsqueeze(-1)
                eval_loss = self.mse(target_points, pred_points)

                wandb.log({"loss": eval_loss})

        wandb.finish()    


if __name__ == "__main__":
    dt = SelfsupervisedDiffusionTracker()
    dt.train()
    torch.save(dt.model.state_dict(), 'unsupervised.pth')









            






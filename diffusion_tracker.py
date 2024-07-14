import os
import yaml
import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from algorithms.feature_extraction_loading import FeatureDataset, concatenate_video_features
from algorithms.utils import read_config_file, feature_collate_fn
from evaluation.evaluation_datasets import compute_tapvid_metrics
from learning_based.model import FeatureDictProcessor, TrackingModel
from learning_based.residual_block import ResidualFeatureBlock
import time
import torch.nn.functional as func
from algorithms.extract_optical_flow import extract_optical_flow_pairs, preprocess_video, plot_images_with_points
from info_nce import InfoNCE
from evaluation.visualization import visualize_heatmaps
from math import ceil
from random import randint
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

N_FRAMES = 8
N_POINTS = 512
N_CHANNELS = 64
ACCUM_ITER = 4
BATCH_SIZE = 64
VIDEO_IDX = 0
FEATURE_TYPE = "better_feature" #features or better_feature

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    # Collect gradients
    gradients = []
    for name, param in named_parameters:
        if param.requires_grad:
            gradients.append((name, param.grad.view(-1).cpu().data.numpy()))
    
    # Plot gradients
    fig, axes = plt.subplots(len(gradients), 1, figsize=(10, len(gradients) * 2))
    if len(gradients) == 1:
        axes = [axes]
    
    for ax, (name, grad) in zip(axes, gradients):
        ax.hist(grad, bins=50)
        ax.set_title(f'Gradient Histogram for {name}')
        ax.set_xlabel('Gradient value')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()

    return plt

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
        self.data = dataset[VIDEO_IDX][0]

        self.of_point_pair_path = os.path.join('a_video_dir', 'video_' + str(VIDEO_IDX), 'of_trajectories.pt')

        self.track_model = TrackingModel().to(device)
        self.residual_block = ResidualFeatureBlock(intermediate_channels=[32, 64], n_output_channels=N_CHANNELS).to(device)

        params = list(self.track_model.parameters()) + list(self.residual_block.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config['learning_rate'])        
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, 0.999)

        self.mse = torch.nn.MSELoss()
        self.huber = torch.nn.HuberLoss()
        self.loss_fn_infonce = InfoNCE(negative_mode='paired')

    def contrastive_loss(self, query_points, target_points, features):
        F, C, H, W = features.shape
        N, _ = query_points.shape

        query_features = features[query_points[:,0].to(dtype=int), :, query_points[:,1].to(dtype=int), query_points[:,2].to(dtype=int)]
        target_features = features[target_points[:,0].to(dtype=int), :, target_points[:,1].to(dtype=int), target_points[:,2].to(dtype=int)]

        negative_features_1 = features[target_points[:,0].to(dtype=int)].permute(0, 2, 3, 1)
        mask_1 = torch.ones(N, H, W, dtype=torch.bool, device=features.device)
        mask_1[torch.arange(N), target_points[:,1].to(dtype=int), target_points[:,2].to(dtype=int)] = False
        negative_features_1 = negative_features_1[mask_1].view(N, H * W - 1, C)

        negative_features_2 = features[query_points[:,0].to(dtype=int)].permute(0, 2, 3, 1)
        mask_2 = torch.ones(N, H, W, dtype=torch.bool, device=features.device)
        mask_2[torch.arange(N), query_points[:,1].to(dtype=int), query_points[:,2].to(dtype=int)] = False
        negative_features_2 = negative_features_2[mask_2].view(N, H * W - 1, C)

        #negative_features = torch.cat([negative_features_1, negative_features_2], dim=1)

        loss = (self.loss_fn_infonce(query_features, target_features, negative_features_1) + self.loss_fn_infonce(target_features, query_features, negative_features_2)) * 0.5

        return loss
        
    def loss_of(self, of_point_pairs, features):

        pred_points_1 = self.track_model.forward_skip(features, of_point_pairs[0]) #NxFx2
        pred_points_2 = self.track_model.forward_skip(features, of_point_pairs[1])
        #of_point_pairs[1][:,0] N

        N, F, _ = pred_points_1.shape

        #pred_endpoints = pred_points[:, of_point_pairs[1][:, 0].long()]
        pred_endpoints_1 = torch.gather(pred_points_1, dim=1, index=of_point_pairs[1][:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2).long()).squeeze()
        pred_endpoints_2 = torch.gather(pred_points_2, dim=1, index=of_point_pairs[0][:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2).long()).squeeze()
        #print(pred_endpoints)
        #print(of_point_pairs[1][:, 1:])
        loss_of = self.huber(pred_endpoints_1, of_point_pairs[1][:, 1:]) + self.huber(pred_endpoints_2, of_point_pairs[0][:, 1:])

        return loss_of

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

    def evaluate(self, query_points, target_points, occluded, features, video_tensor):

        batch_query_point = []
        batch_gt_occluded = []
        batch_gt_point = []
        batch_pred_point = []

        self.residual_block.eval()
        self.track_model.eval()
        
        with torch.no_grad():
            refined_features = features + self.residual_block(video_tensor)
            heatmaps = self.track_model.heatmap_generator.generate(refined_features, query_points)
            pred_points = self.track_model.heatmap_processor.predictions_from_heatmap(heatmaps)

        wandb.log({"chart1": visualize_heatmaps(video_tensor, heatmaps[4].cpu(), pred_points[4], target_points[4])})
        wandb.log({"chart2": visualize_heatmaps(video_tensor, heatmaps[5].cpu(), pred_points[5], target_points[5])})
        wandb.log({"chart3": visualize_heatmaps(video_tensor, heatmaps[6].cpu(), pred_points[6], target_points[6])})
        wandb.log({"chart4": visualize_heatmaps(video_tensor, heatmaps[7].cpu(), pred_points[7], target_points[7])})
        plt.close()

        batch_pred_point.append(pred_points.cpu().numpy())
        batch_query_point.append(query_points.cpu().numpy())
        batch_gt_point.append(target_points.cpu().numpy())
        batch_gt_occluded.append(occluded.cpu().numpy())

        metrics = compute_tapvid_metrics(query_points=np.array(batch_query_point), gt_occluded=np.array(batch_gt_occluded), gt_tracks=np.array(batch_gt_point), pred_occluded=np.array(batch_gt_occluded), pred_tracks=np.array(batch_pred_point), query_mode='strided')
        print(metrics)

        self.residual_block.train()
        self.track_model.train()

        return metrics['average_pts_within_thresh']

    def get_of_point_pair_batch(self, of_point_pair,  batch_size=32, shuffle=True, drop_last=False):
        """
        Yields a tuple containing one batch of query_points, target_points, occluded, trackgroup
        """
        num_points = of_point_pair[0].shape[0]

        if shuffle:
            permutation = np.random.permutation(num_points)
            of_point_pair = (of_point_pair[0][permutation], of_point_pair[1][permutation])

        if drop_last:
            num_batches = num_points // batch_size
        else:
            num_batches = ceil(num_points / batch_size)
        
        for i in range(num_batches):
            start = i*batch_size
            end = min((i+1)*batch_size, num_points)

            yield (
                of_point_pair[0][start:end], 
                of_point_pair[1][start:end],
            )

    def train(self):

        wandb.init(entity=self.config['wandb']['entity'],
          project=self.config['wandb']['project'],
          mode="disabled",
          config=self.config)
        
        
        features = self.data[FEATURE_TYPE].to(device)
        target_points = torch.tensor(self.data["target_points"][..., [1, 0]], dtype=torch.float32, device=device)[0]
        occluded = torch.tensor(self.data["occluded"], dtype=torch.float32, device=device)[0]
        query_points = torch.tensor(self.data["query_points"], dtype=torch.float32, device=device)[0]
        
        video = self.data["video"][0] / 255.0
        video_tensor = torch.tensor(video, device=device, dtype=torch.float32).permute(0, 3, 1, 2)

        features = features[:N_FRAMES, :N_CHANNELS]
        video_tensor = video_tensor[:N_FRAMES]
        target_points = target_points[:, :N_FRAMES]
        occluded = occluded[:, :N_FRAMES]

        query_points = query_points[query_points[:, 0] < N_FRAMES]
        target_points = target_points[:query_points.shape[0]]
        occluded = occluded[:query_points.shape[0]]

        of_point_pairs = torch.load(self.of_point_pair_path).to(device)
        of_point_pairs = of_point_pairs[of_point_pairs[:, 3] < N_FRAMES]

        permutation = np.random.permutation(of_point_pairs.shape[0])
        of_point_pairs = of_point_pairs[permutation]
        of_point_pairs = of_point_pairs[:N_POINTS]

        print(of_point_pairs.shape)

#For visually inspecting OF point pairs
#        video_tensor = video_tensor.cpu()
#        of_point_pairs = of_point_pairs.cpu()
#        while True:
#            idx = randint(0, of_point_pairs.shape[0])
#            images = torch.cat((video_tensor[of_point_pairs[idx][0].long()].unsqueeze(0), video_tensor[of_point_pairs[idx][3].long()].unsqueeze(0)), dim=0)
#
#            point_pair = [(of_point_pairs[idx][0], of_point_pairs[idx][1], of_point_pairs[idx][2]), (of_point_pairs[idx][3], of_point_pairs[idx][4], of_point_pairs[idx][5])]
#
#            plot_images_with_points(images, point_pair)

        of_point_pairs = (of_point_pairs[:,:3], of_point_pairs[:, 3:])

        total_iterations = self.config["total_iterations"]

        #for iter in tqdm(range(total_iterations), desc="Train iteration"):
        for iter in range(total_iterations):

            ## Train
            self.track_model.train()
            self.residual_block.train()
            
            running_loss = 0

#            residual_features = self.residual_block(video_tensor)
#            refined_features = features + residual_features
#            pred_points = self.track_model.forward_skip(refined_features, query_points)
#
#            print(torch.cat([pred_points, target_points], dim=-1))
#
#            running_loss = self.huber(pred_points, target_points)
#
#            running_loss.backward()
#            self.optimizer.step()
#            self.scheduler.step()
#            self.optimizer.zero_grad()

            for batch_idx, of_point_pair_batch in enumerate(self.get_of_point_pair_batch(of_point_pairs, batch_size=BATCH_SIZE)):

                with torch.set_grad_enabled(True):
                    refined_features = features + self.residual_block(video_tensor)

                    loss = self.loss_of(of_point_pair_batch, refined_features)
                    loss += 0.001 * self.contrastive_loss(of_point_pair_batch[0], of_point_pair_batch[1], refined_features)
                    #if(iter > 70):
                    #    loss += 0.1 * self.loss_of(of_point_pair_batch, refined_features)

                    #pred_points = self.track_model.forward_skip(refined_features, query_points)
                    #running_loss = self.huber(pred_points, target_points)

                    #loss = loss / ACCUM_ITER

                    loss.backward()

                    # weights update
                    #if ((batch_idx + 1) % ACCUM_ITER == 0) or (batch_idx + 1 == of_point_pairs[0].shape[0] // BATCH_SIZE):
                        #wandb.log({"gradients": plot_grad_flow(self.residual_block.named_parameters())})
                        #plt.close()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                norm_res = 0
                norm_track = 0
                #norm_res = torch.nn.utils.clip_grad_norm_(self.residual_block.parameters(), 100)
                #norm_track = torch.nn.utils.clip_grad_norm_(self.track_model.parameters(), 100)
                #self.optimizer.step()
                #self.scheduler.step()

                running_loss += loss

            print(f"{iter}: loss: {running_loss:.4f} | norm-residual: {norm_res:.4f} | norm-track: {norm_track:.4f}")
            
            wandb.log({"running_loss": running_loss})

            if iter % self.config['eval_freq'] == 0:
                eval_error = self.evaluate(query_points, target_points, occluded, features, video_tensor)
                wandb.log({"evaluation": eval_error})

        wandb.finish()    


if __name__ == "__main__":

    dt = SelfsupervisedDiffusionTracker()
    dt.train()
    torch.save(dt.model.state_dict(), 'unsupervised.pth')
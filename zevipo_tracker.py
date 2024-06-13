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
from torch.utils.tensorboard import SummaryWriter
from learning_based.weighted_features_tracker import WeightedFeaturesTracker, WeightedHeatmapsTracker
from torch.cuda.amp import GradScaler, autocast
from math import ceil
from torch.autograd import gradcheck

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


from torchvision.transforms.functional import resize 
from algorithms.heatmap_generator import HeatmapGenerator
from algorithms.zero_shot_tracker import ZeroShotTracker

w = (
    torch.tensor([1, 1, 1, 1], dtype=torch.float64, requires_grad=True),
    torch.tensor([1, 1, 1, 1], dtype=torch.float64, requires_grad=True),
    torch.tensor([1], dtype=torch.float64, requires_grad=True),
    torch.tensor([1, 1, 1, 1], dtype=torch.float64, requires_grad=True),
)

def model_forward(weights):
    heatmap_generator = HeatmapGenerator()
    tracker = ZeroShotTracker()

    # dict
    # query

    hmps_list = []

    for i, block in enumerate(inputs):
                
        resized_features = resize(block,  (256, 256))

        hmps = heatmap_generator.generate(resized_features, query_points)

        # Apply weight to heatmap
        hmps *= [block_name][i]

        hmps_list.append(hmps)
            
    # Sum up to single heatmap
    concat_heatmaps = torch.sum(torch.stack(hmps_list), dim=0)

    tracks = tracker.track(concat_heatmaps)

    return tracks

    

class ZeViPo():
    def __init__(self, config_file: str):

        self.config = read_config_file(config_file)

        self.dataset = FeatureDataset(feature_dataset_path=self.config['dataset_dir'])
        self.dataloader = DataLoader(self.dataset, self.config['batch_size'], shuffle=True, collate_fn=feature_collate_fn)

        #self.model = WeightedFeaturesTracker(next(iter(self.dataloader))[0]["features"]).to(device).half()
        self.model = WeightedHeatmapsTracker(next(iter(self.dataloader))[0]["features"]).to(device)

        self.loss_fn = torch.nn.HuberLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scaler = GradScaler()

        self.epochs = self.config['epochs']

    def get_query_batch(self, query_points, target_points, occluded, trackgroup, batch_size=8, shuffle=True, drop_last=False):
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

    def train_one_epoch(self, epoch_index):
        for i, data in enumerate(self.dataloader):
            data = data[0]

            ### Overfitting to single data point
            data["query_points"] = data["query_points"][:,:1,:]
            data["target_points"] = data["target_points"][:,:1,:]
            data["occluded"] = data["occluded"][:,:1]
            data["trackgroup"] = data["trackgroup"][:,:1]

            # print(data["query_points"].shape)
            # print(data["target_points"].shape)
            # print(data["occluded"].shape)
            # print(data["trackgroup"].shape)
            ###

            feature_dict = data['features']
            for block_name, block_feat_list in feature_dict.items():
                for i in range(len(block_feat_list)):
                    feature_dict[block_name][i] = feature_dict[block_name][i].to(dtype=torch.float32).to(device)

            accumulated_loss = 0
            loop_count = 0
            for query_batch in self.get_query_batch(data["query_points"][0], data["target_points"][0], data["occluded"][0], data["trackgroup"][0]):
                loop_count += 1

                query_points, target_points, occluded, trackgroup = query_batch

                query_points = torch.tensor(query_points, dtype=torch.float32, device=device)
                target_points = torch.tensor(target_points[..., [1, 0]], dtype=torch.float32, device=device)
                occluded = torch.tensor(occluded, dtype=torch.float32, device=device)
                trackgroup = torch.tensor(trackgroup, dtype=torch.float32, device=device)

                self.optimizer.zero_grad()

                #with autocast():
                #    pred_points = self.model(feature_dict, query_points)
                #    losses = self.loss_fn(target_points, pred_points)
                #    loss = torch.mean(losses * (1 - occluded).unsqueeze(-1))

                pred_points = self.model(feature_dict, query_points)
                losses = self.loss_fn(target_points, pred_points)
                loss = torch.mean(losses * (1 - occluded).unsqueeze(-1))

                loss.backward()
                #self.optimizer.step()

                #self.scaler.scale(loss).backward()
                #self.scaler.step(self.optimizer)
                #self.scaler.update()

                for param in self.model.parameters():
                    print(param)
                    print(param.grad)

                test = gradcheck(self.model_forward, (feature_dict, query_points))
                print("gradien check: " + test)


                accumulated_loss += loss.item()
            
            print(accumulated_loss/loop_count)

            #wandb.log({"Loss/train": accumulated_loss/loop_count})


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
        #wandb.init(entity=self.config['wandb']['entity'],
        #           project=self.config['wandb']['project'],
        #           config=self.config)
        
        for epoch in tqdm(range(self.epochs)):

            self.model.train(True)
            self.train_one_epoch(epoch)
            continue

            self.model.eval()
            mean_tracking_accuracy = self.eval_random()

            logger.add_scalar('Acc/train', mean_tracking_accuracy, epoch)

            logger.flush()
        
        #wandb.finish()


if __name__ == '__main__':
    tracker = ZeViPo('configs/config.yml')

    tracker.train()
    
    #print(tracker.eval_random())
import torch
from torchvision.transforms.functional import resize 
from algorithms.heatmap_generator import HeatmapGenerator
from algorithms.zero_shot_tracker import ZeroShotTracker
from algorithms.feature_extraction_loading import concatenate_video_features

class LearnUpsampleTracker(torch.nn.Module):
    def __init__(self, feature_dict):
        super().__init__()

        self.heatmap_generator = HeatmapGenerator()
        self.tracker = ZeroShotTracker()

        self.softmax = torch.nn.Softmax()

        # Create upsample layers for upblock and decoder_block
        self.upsamples_upblock = torch.nn.ModuleList(
            torch.nn.ConvTranspose2d(10, 10, 32),
            torch.nn.ConvTranspose2d(10, 10, 16),
            torch.nn.ConvTranspose2d(10, 10, 8),
            torch.nn.ConvTranspose2d(10, 10, 8),
        )

        self.upsamples_decoderblock = torch.nn.ModuleList(
            torch.nn.ConvTranspose2d(10, 10, 4),
            torch.nn.ConvTranspose2d(10, 10, 2),
            torch.nn.ConvTranspose2d(10, 10, 1),
            torch.nn.ConvTranspose2d(10, 10, 1),
        )

        

    def forward(self, feature_dict, query_points):

        upsampled_features = []

        for block_name, block_feature_list in feature_dict:
            for i, feat_map in enumerate(block_feature_list):
                if block_name == "up_block":
                    upsampled_features.append(self.upsamples_upblock[i](feat_map))
                elif block_name == "decoder_block":
                    upsampled_features.append(self.upsamples_decoderblock[i](feat_map))
                else:
                    pass
        
        features = torch.concat(upsampled_features)

        hmps = self.heatmap_generator.generate(features, query_points)

        tracks = self.tracker.track(hmps)

        return tracks

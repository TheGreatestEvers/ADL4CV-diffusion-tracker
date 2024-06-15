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

        self.relu = torch.nn.ReLU()

        # Create upsample layers
        #self.upsamples_downblock = torch.nn.ModuleList([
        #    torch.nn.ConvTranspose2d(10, 10, 16, stride=16),
        #    torch.nn.ConvTranspose2d(10, 10, 32, stride=32),
        #    torch.nn.ConvTranspose2d(10, 10, 64, stride=64),
        #    torch.nn.ConvTranspose2d(10, 10, 64, stride=64),
        #])

        #self.upsamples_midblock = torch.nn.ModuleList([
        #    torch.nn.ConvTranspose2d(10, 10, 64, stride=64),
        #])

        #self.upsamples_upblock = torch.nn.ModuleList([
        #    torch.nn.ConvTranspose2d(10, 10, 32, stride=32),
        #    torch.nn.ConvTranspose2d(10, 10, 16, stride=16),
        #    torch.nn.ConvTranspose2d(10, 10, 8, stride=8),
        #    torch.nn.ConvTranspose2d(10, 10, 8, stride=8),
        #])

        #self.upsamples_decoderblock = torch.nn.ModuleList([
        #    torch.nn.ConvTranspose2d(10, 10, 4, stride=4),
        #    torch.nn.ConvTranspose2d(10, 10, 2, stride=2),
        #    torch.nn.ConvTranspose2d(10, 10, 1, stride=1),
        #    torch.nn.ConvTranspose2d(10, 10, 1, stride=1),
        #])

        self.up_8_16 = torch.nn.ConvTranspose2d(10, 10, 2, stride=2)
        self.up_16_32 = torch.nn.ConvTranspose2d(20, 20, 2, stride=2)
        self.up_32_64 = torch.nn.ConvTranspose2d(40, 40, 2, stride=2)
        self.up_64_128 = torch.nn.ConvTranspose2d(50, 50, 2, stride=2)
        self.up_128_256 = torch.nn.ConvTranspose2d(60, 60, 2, stride=2)



        

    def forward(self, feature_dict, query_points):

        #upsampled_features = []

        #for block_name, block_feature_list in feature_dict.items():
        #    for i, feat_map in enumerate(block_feature_list):
        #        if block_name == "up_block":
        #            upsampled_features.append(self.relu(self.upsamples_upblock[i](feat_map)))
        #            #print(feat_map.shape)
        #            #print(self.upsamples_upblock[i](feat_map).shape)
        #        elif block_name == "decoder_block":
        #            upsampled_features.append(self.relu(self.upsamples_decoderblock[i](feat_map)))
        #            #print(feat_map.shape)
        #            #print(self.upsamples_decoderblock[i](feat_map).shape)
        #        elif block_name == "down_block":
        #            upsampled_features.append(self.relu(self.upsamples_downblock[i](feat_map)))
        #            #print(feat_map.shape)
        #            #print(self.upsamples_decoderblock[i](feat_map).shape)
        #        elif block_name == "mid_block":
        #            upsampled_features.append(self.relu(self.upsamples_midblock[i](feat_map)))
        #            #print(feat_map.shape)
        #            #print(self.upsamples_decoderblock[i](feat_map).shape)
        #        else:
        #            pass

        f16 = self.relu(self.up_8_16(feature_dict["up_block"][0] + feature_dict["down_block"][1]))

        f16 = f16 + feature_dict["down_block"][0]
        f16 = torch.cat((f16, feature_dict["up_block"][1]), dim=1)
        f32 = self.relu(self.up_16_32(f16))

        f32 = torch.cat((f32, feature_dict["up_block"][2], feature_dict["up_block"][3]), dim=1)
        f64 = self.relu(self.up_32_64(f32))

        f64 = torch.cat((f64, feature_dict["decoder_block"][0]), dim=1)
        f128 = self.relu(self.up_64_128(f64))

        f128 = torch.cat((f128, feature_dict["decoder_block"][1]), dim=1)
        f256 = self.relu(self.up_128_256(f128))

        f256 = torch.cat((f256, feature_dict["decoder_block"][2], feature_dict["decoder_block"][2]), dim=1)

        hmps = self.heatmap_generator.generate(f256, query_points)

        tracks = self.tracker.track(hmps)

        return tracks

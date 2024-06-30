import torch
import torch.nn.functional as F
from algorithms.heatmap_generator import HeatmapGenerator
from algorithms.heatmap_processor import HeatmapProcessor

class FeatureDictProcessor(torch.nn.Module):
    """
    Process feature dictionary and return one refined feature tensor.
    """

    def __init__(self, feature_dict):
        super().__init__()

        self.relu = torch.nn.ReLU()

        n_channels = 32

        # 1D Conv layers
        self.conv1d_up8 = torch.nn.Conv2d(feature_dict["up_block"][0].shape[1], n_channels, 1)
        self.conv1d_up16 = torch.nn.Conv2d(feature_dict["up_block"][1].shape[1], n_channels, 1)
        self.conv1d_up32_1 = torch.nn.Conv2d(feature_dict["up_block"][2].shape[1], n_channels, 1)
        self.conv1d_up32_2 = torch.nn.Conv2d(feature_dict["up_block"][3].shape[1], n_channels, 1)

        self.conv1d_down16 = torch.nn.Conv2d(feature_dict["down_block"][0].shape[1], n_channels, 1)
        self.conv1d_down8 = torch.nn.Conv2d(feature_dict["down_block"][1].shape[1], n_channels, 1)
        self.conv1d_down4_1 = torch.nn.Conv2d(feature_dict["down_block"][2].shape[1], n_channels, 1)
        self.conv1d_down4_2 = torch.nn.Conv2d(feature_dict["down_block"][3].shape[1], n_channels, 1)

        self.conv1d_mid4 = torch.nn.Conv2d(feature_dict["mid_block"][0].shape[1], n_channels, 1)

        self.conv1d_dec64 = torch.nn.Conv2d(feature_dict["decoder_block"][0].shape[1], n_channels, 1)
        self.conv1d_dec128 = torch.nn.Conv2d(feature_dict["decoder_block"][1].shape[1], n_channels, 1)
        self.conv1d_dec256_1= torch.nn.Conv2d(feature_dict["decoder_block"][2].shape[1], n_channels, 1)
        self.conv1d_dec256_2 = torch.nn.Conv2d(feature_dict["decoder_block"][3].shape[1], n_channels, 1)


        # Upsample layers
        # self.upsample_4_to_8 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)
        # self.upsample_8_to_16 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)
        # self.upsample_16_to_32 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)
        # self.upsmaple_32_to_64 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)
        # self.upsample_64_to_128 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)
        # self.upsample_128_to_256 = torch.nn.ConvTranspose2d(n_channels, n_channels, 2, stride=2)

        # Refining Conv layers
        self.refine_conv1 = torch.nn.Conv3d(n_channels, 64, 3, padding=1) # Make this maintain spatial size
        self.refine_conv2 = torch.nn.Conv3d(64, 64, 3, padding=1) # Make this maintain spatial size


    def forward(self, feature_dict, query_points):

        ### Process features

        # Spatial: 4
        features = self.conv1d_mid4(feature_dict["mid_block"][0]) \
            + self.conv1d_down4_2(feature_dict["down_block"][3]) \
            + self.conv1d_down4_1(feature_dict["down_block"][2])

        # Spatial 8        
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        #features = self.upsample_4_to_8(features)

        features = features \
            + self.conv1d_up8(feature_dict["up_block"][0]) \
            + self.conv1d_down8(feature_dict["down_block"][1]) 
        
        # Spatial 16
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # features = self.upsample_8_to_16(features)

        features = features \
            + self.conv1d_up16(feature_dict["up_block"][1]) \
            + self.conv1d_down16(feature_dict["down_block"][0])
        
        # Spatial 32
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # features = self.upsample_16_to_32(features)

        features = features \
            + self.conv1d_up32_1(feature_dict["up_block"][2]) \
            + self.conv1d_up32_2(feature_dict["up_block"][3])
        
        # Spatial 64
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # features = self.upsample_32_to_64(features)

        features = features \
            + self.conv1d_dec64(feature_dict["decoder_block"][0])
        
        # Spatial 128
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # features = self.upsample_64_to_128(features)

        features = features \
            + self.conv1d_dec128(feature_dict["decoder_block"][1])
        
        # Spatial 256
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # features = self.upsample_128_to_256(features)

        features = features \
            + self.conv1d_dec256_1(feature_dict["decoder_block"][2]) \
            + self.conv1d_dec256_2(feature_dict["decoder_block"][3])
        
        # Refine
        features = torch.permute(features, (1, 0, 2, 3)) # C, F, H, W
        features = self.refine_conv1(features)
        features = self.relu(features)
        features = self.refine_conv2(features)
        features = self.relu(features)
        features = torch.permute(features, (1, 0, 2, 3)) # F, C, H, W

        return features

class TrackingModel(torch.nn.Module):
    """
    Stepwise tracking
    """

    def __init__(self) -> None:
        super().__init__()

        self.heatmap_generator = HeatmapGenerator()
        self.heatmap_processor = HeatmapProcessor()

    def forward(self, features, query):
        """
        Features: FxCxHxW
        Query: 1x3
        """

        # Throw away frames prior to query
        t = query[0, 0].long()

        features = features[t:]
        query[0, 0] -= t # Query is in first frame of modified feature tensor

        F = features.shape[0]

        points = torch.zeros(F, 2)
        points[0] = query[0, 0]

        for i in range(F-1):
            
            feat_t0 = features[i]
            feat_t1 = features[i+1]

            feat_consecutive_frames = torch.cat((feat_t0, feat_t1))

            heatmap = self.heatmap_generator.generate(feat_consecutive_frames, query)

            point, _ = self.heatmap_processor.predictions_from_heatmap(heatmap)[1]

            points[i+1] = point

            # Set new query point to last predicted point
            query[0, 0] = point
        
        return points




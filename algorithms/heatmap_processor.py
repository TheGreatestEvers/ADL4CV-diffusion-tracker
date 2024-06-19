import torch

class HeatmapProcessor(torch.nn.Module):
    """
    Process heatmap and obtain estimation for tracks and occluded points
    """
    
    def __init__(self) -> None:
        super().__init__()

        self.argmax_radius = 34
        self.softmax = torch.nn.Softmax(dim=-1)

        self.relu = torch.nn.ReLU()

        self.heatmap_processing_layers = torch.nn.ModuleDict({
            'hid1': torch.nn.Conv2d(
                in_channels=1,  # Adjust this according to your input channel size
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            'hid2': torch.nn.Conv2d(
                in_channels=16,  # Adjust this according to your input channel size
                out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            'hid3': torch.nn.Conv2d(
                in_channels=1,  # Adjust this according to your input channel size
                out_channels=32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            'hid4': torch.nn.Linear(1, 16),  # Adjust input size according to your feature map size
            'occ_out': torch.nn.Linear(16, 1),  # Adjust input size according to your feature map size
            'regression_hid': torch.nn.Linear(32, 128),  # Adjust input size according to your feature map size
            'regression_out': torch.nn.Linear(128, 2)
        })

    def predictions_from_heatmap(self, heatmaps):
        """
        Get track estimates in each frame.

        Args:
            heatmaps: Heatmaps with Dimensions: [NumPoints, Frames, Height, Width]

        Returns:
            tracks: Tensor containing xy coordinate of each estimated point in each frame. Dimensions: [NumPoints, Frames, 2]
        """

        N, F, H, W = heatmaps.shape

        # Add channel dim and reshape to N*F, C, H, W
        heatmaps = heatmaps.unsqueeze(2).view(N*F, 1, H, W)

        processed_heatmaps = self.heatmap_processing_layers["hid1"](heatmaps)
        processed_heatmaps = self.relu(processed_heatmaps)

        assert processed_heatmaps.shape[-1] == heatmaps.shape[-1]

        # Position inference
        position_heatmaps = self.heatmap_processing_layers["hid2"](processed_heatmaps)
        position_heatmaps_shape = position_heatmaps.shape
        position_heatmaps = torch.flatten(position_heatmaps, start_dim=2)
        position_heatmaps = self.softmax(position_heatmaps)

        argmax_flat = torch.argmax(position_heatmaps.squeeze(), dim=-1)
        argmax_indices = torch.stack((argmax_flat // position_heatmaps_shape[-1], argmax_flat % position_heatmaps_shape[-1]), dim=-1)

        points = self.soft_argmax(position_heatmaps.view(position_heatmaps_shape).squeeze(), argmax_indices)
        points = points.view(N, F, -1)

        # Occlusion inference
        occlusions = torch.zeros(1)


        

        # tracks = torch.zeros([N, F, 2], dtype=torch.float32, device=heatmaps.device)

        # for i, hmp in enumerate(heatmaps):

        #     # Perform softmax on heatmaps
        #     hmp_softmax = self.softmax(hmp.view(F, -1)) # Shape F, H*W

        #     # Compute argmax indices
        #     argmax_flat = torch.argmax(hmp_softmax, dim=1)
        #     argmax_indices = torch.stack((argmax_flat // hmp.shape[-1], argmax_flat % hmp.shape[-1]), dim=-1)

        #     # Get soft argmax indices
        #     tracks[i] = self.soft_argmax(hmp, argmax_indices)
        
        return points, occlusions

    def soft_argmax(self, heatmap, argmax_indices):
        """
        Computes soft argmax.

        Args:
            heatmap: Tensor with shape [Frames, Height, Width]
            argmax_indices: Hard argmax indices. Tensor with shape [Frames, 2]
        
        Returns:
            Soft argmax indices. Tensor with shape [Frames, 2]
        """

        F, H, W = heatmap.shape

        # Create grid of shape FxHxWx2
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid_y = grid_y.unsqueeze(0).expand(F, -1, -1).float()
        grid_x = grid_x.unsqueeze(0).expand(F, -1, -1).float()
        grid = torch.stack((grid_y, grid_x), dim=-1).to(heatmap.device)

        # Generate mask of a circle of radius radius around the argmax
        mask = torch.norm((grid - argmax_indices.unsqueeze(1).unsqueeze(2)), dim=-1) <= self.argmax_radius # shape (B, H, W)

        # Apply mask and get sums
        heatmap = heatmap * mask
        hm_sum = torch.sum(heatmap, dim=(1, 2)) # F
        hm_zero_indices = hm_sum < 1e-8
        
        # for numerical stability
        if sum(hm_zero_indices) > 0:
            uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
            heatmap[hm_zero_indices] += uniform_w[:, None, None]
            heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
            hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        points = torch.sum(grid.float() * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (F, 2)
        
        return points

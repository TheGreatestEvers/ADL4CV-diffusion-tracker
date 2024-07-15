import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HeatmapProcessor(torch.nn.Module):
    """
    Process heatmap and obtain estimation for tracks and occluded points
    """
    
    def __init__(self) -> None:
        super().__init__()

        self.argmax_radius = 35
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

        # Flatten the heatmaps for argmax calculation
        heatmaps_flat = heatmaps.view(N*F, H*W)
        argmax_flat = torch.argmax(heatmaps_flat, dim=-1)

        # Convert flat indices to (x, y) coordinates
        argmax_indices = torch.stack((argmax_flat // W, argmax_flat % W), dim=-1)

        # Add channel dim and reshape to N*F, C, H, W
        heatmaps = heatmaps.unsqueeze(2).view(N*F, 1, H, W)

        heatmaps = self.heatmap_processing_layers["hid1"](heatmaps)
        heatmaps = self.relu(heatmaps)

        # Position inference
        heatmaps = self.heatmap_processing_layers["hid2"](heatmaps)

        heatmaps = heatmaps.squeeze()
        heatmaps = heatmaps.view(-1, H*W)
        heatmaps = self.softmax(heatmaps)

        #argmax_flat = torch.argmax(heatmaps, dim=-1)
        #argmax_indices = torch.stack((argmax_flat // W, argmax_flat % W), dim=-1)

        points = self.soft_argmax(heatmaps.view(-1, H, W), argmax_indices)
        
        return points.view(N, F, -1)



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
        grid_y, grid_x = torch.meshgrid((torch.arange(H, device=device), torch.arange(W, device=device)))
        grid_y = grid_y.unsqueeze(0).expand(F, -1, -1).float()
        grid_x = grid_x.unsqueeze(0).expand(F, -1, -1).float()
        grid = torch.stack((grid_y, grid_x), dim=-1)

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

    def soft_argmax_heatmap(
        self,
        heatmaps: torch.Tensor,
        threshold: float = 5.0,
    ) -> torch.Tensor:
        """Computes the soft argmax of a heatmap.

        Finds the argmax grid cell, and then returns the average coordinate of
        surrounding grid cells, weighted by the softmax.

        Args:
            softmax_val: A heatmap of shape [N, F, height, width], containing all positive
            values summing to 1 across the entire grid.
            threshold: The radius of surrounding cells to consider when computing the
            average.

        Returns:
            The soft argmax, which is a single point [x,y] in grid coordinates.
        """
        num_points, frames, height, width = heatmaps.shape

        points = torch.zeros(num_points, frames, 2, device=device)

        y, x = torch.meshgrid(torch.arange(height).to(device), torch.arange(width).to(device))
        
        for n in range(num_points):
            for f in range(frames):
                hmp = heatmaps[n, f]
                coords = torch.stack([x + 0.5, y + 0.5], dim=-1)
                argmax_pos = torch.argmax(hmp.view(-1))
                pos = coords.view(-1, 2)[argmax_pos].view(1, 1, 2)
                valid = torch.sum((coords - pos) ** 2, dim=-1, keepdim=True) < threshold ** 2
                weighted_sum = torch.sum(coords * valid * hmp.unsqueeze(-1), dim=(0, 1))
                sum_of_weights = torch.maximum(torch.sum(valid * hmp.unsqueeze(-1), dim=(0, 1)), torch.tensor(1e-12))
                points[n, f] = weighted_sum / sum_of_weights
        
        return points

    def heatmaps_to_points(
        self,
        all_pairs_softmax: torch.Tensor,
        image_shape: torch.Size = torch.tensor([256, 256]),
        threshold: float = 5.0
    ) -> torch.Tensor:
        """Given a batch of heatmaps, compute a soft argmax.

        If query points are given, constrain that the query points are returned
        verbatim.

        Args:
            all_pairs_softmax: A set of heatmaps, of shape [num_points, time,
            height, width].
            image_shape: The shape of the original image that the feature grid was
            extracted from.  This is needed to properly normalize coordinates.
            threshold: Threshold for the soft argmax operation.
            query_points (optional): If specified, we assume these points are given as
            ground truth and we reproduce them exactly.  This is a set of points of
            shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
            raster coordinates.

        Returns:
            Predicted points, of shape [batch, num_points, time, 2], where each point is
            [x, y] in raster coordinates.  These are the result of a soft argmax except
            where the query point is specified, in which case the query points are
            returned verbatim.
        """
        # soft_argmax_heatmap operates over a single heatmap. We map it across
        # batch, num_points, and frames.
        vmap_sah = torch.vmap(self.soft_argmax_heatmap, in_dims=(0, None))
        for _ in range(2):
            vmap_sah = torch.vmap(vmap_sah, in_dims=(0, None))

        out_points = vmap_sah(all_pairs_softmax, threshold)
        feature_grid_shape = all_pairs_softmax.shape[2:]

        # Note: out_points is now [x, y]; we need to divide by [width, height].
        out_points = out_points / torch.tensor([feature_grid_shape[2], feature_grid_shape[1]])

        return out_points
    

class FixedHeatmapProcessor(torch.nn.Module):
    """
    Process heatmap and obtain estimation for tracks and occluded points
    """
    
    def __init__(self) -> None:
        super().__init__()

        self.argmax_radius = 34
        self.softmax = torch.nn.Softmax(dim=-1)

    def predictions_from_heatmap(self, heatmaps):
        """
        Get track estimates in each frame.

        Args:
            heatmaps: Heatmaps with Dimensions: [NumPoints, Frames, Height, Width]

        Returns:
            tracks: Tensor containing xy coordinate of each estimated point in each frame. Dimensions: [NumPoints, Frames, 2]
        """

        N, F, H, W = heatmaps.shape

        heatmaps = torch.flatten(heatmaps, start_dim=2)
        heatmaps = self.softmax(heatmaps)

        argmax_flat = torch.argmax(heatmaps.squeeze(), dim=-1)
        argmax_indices = torch.stack((argmax_flat // W, argmax_flat % W), dim=-1)

        points = self.soft_argmax(heatmaps.view(N*F, H, W), argmax_indices.view(N*F, 2))
        points = points.view(N, F, -1)

        # Occlusion inference
        occlusions = torch.zeros(1)
        
        return points, occlusions

    def soft_argmax(self, heatmap, argmax_indices):
        """
        Computes soft argmax.

        Args:
            heatmap: Tensor with shape [Points*Frames, Height, Width]
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



if __name__ == "__main__":
    processor = HeatmapProcessor()

    test_hmp = torch.zeros(1, 1, 256, 256)
    test_hmp[0, 0, 120, 130] = 1
    point = processor.predictions_from_heatmap(test_hmp.to(device))
    print(point)
    print(point.shape)
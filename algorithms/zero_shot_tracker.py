import torch

# copied from OmniMotion
# def gen_grid(h_start, w_start, h_end, w_end, step_h, step_w, device, normalize=False, homogeneous=False):
#     """Generate a grid of coordinates in the image frame.
#     Args:
#         h, w: height and width of the grid.
#         device: device to put the grid on.
#         normalize: whether to normalize the grid coordinates to [-1, 1].
#         homogeneous: whether to return the homogeneous coordinates. homogeneous coordinates are 3D coordinates.
#     Returns:"""
#     if normalize:
#         lin_y = torch.linspace(-1., 1., steps=h_end, device=device)
#         lin_x = torch.linspace(-1., 1., steps=w_end, device=device)
#     else:
#         lin_y = torch.arange(h_start, h_end, step=step_h, device=device)
#         lin_x = torch.arange(w_start, w_end, step=step_w, device=device)
#     grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
#     grid = torch.stack((grid_x, grid_y), -1)
#     if homogeneous:
#         grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
#     return grid  # [h, w, 2 or 3]

class ZeroShotTracker:
    """
    Abstract trajectory of target point from precalculated heatmap
    """
    
    def __init__(self) -> None:

        self.argmax_radius = 24
        self.softmax = torch.nn.Softmax(dim=-1)

    def track(self, heatmaps):
        """
        Get track estimates in each frame.

        Args:
            heatmaps: Heatmaps with Dimensions: [NumPoints, Frames, Height, Width]

        Returns:
            tracks: Tensor containing xy coordinate of each estimated point in each frame. Dimensions: [NumPoints, Frames, 2]
        """

        N, F, H, W = heatmaps.shape

        tracks = torch.zeros([N, F, 2], device=heatmaps.device)

        for i, hmp in enumerate(heatmaps):

            # Perform softmax on heatmaps
            hmp_softmax = self.softmax(hmp.view(F, -1)) # Shape F, H*W

            # Compute argmax indices
            argmax_flat = torch.argmax(hmp_softmax, dim=1)
            argmax_indices = torch.stack((argmax_flat // hmp.shape[-1], argmax_flat % hmp.shape[-1]), dim=-1)

            # Get soft argmax indices
            tracks[i] = self.soft_argmax(hmp, argmax_indices)
        
        return tracks

    def soft_argmax(self, heatmap, argmax_indices):
        """
        Computes soft argmax.

        Args:
            heatmap: Tensor with shape [Frames, Height, Width]
            argmax_indices: Hard argmax indices. Tensor with shape [Frames, 2]
        
        Returns:
            Soft argmax indices. Tensor with shape [Frames, 2]
        """

        # # Copied from DinoTracker
        # h_start = self.patch_size // 2
        # w_start = self.patch_size // 2
        # h_end = ((self.video_h - 2 * h_start) // self.step_h) * self.step_h + h_start + math.ceil(self.step_h / 2)
        # w_end = ((self.video_w - 2 * w_start) // self.step_w) * self.step_w + w_start + math.ceil(self.step_w / 2)
        # grid = gen_grid(h_start=h_start, w_start=w_start, h_end=h_end, w_end=w_end, step_h=self.step_h, step_w=self.step_w,
        #                 device=heatmap.device, normalize=False, homogeneous=False) # shape (H, W, 2)
        # grid = grid.unsqueeze(0).repeat(heatmap.shape[0], 1, 1, 1) # stack and repeat grid to match heatmap shape (B, H, W, 2)
        
        # row, col = argmax_indices
        # argmax_coord = torch.stack((col*self.step_w+w_start, row*self.step_h+h_start), dim=-1) # (x,y) coordinates, shape (B, 2)
        
        # # generate a mask of a circle of radius radius around the argmax_coord (B, 2) in heatmap (B, H, W, 2)
        # mask = torch.norm((grid - argmax_coord.unsqueeze(1).unsqueeze(2)).to(torch.float32), dim=-1) <= self.argmax_radius # shape (B, H, W)
        # heatmap = heatmap * mask
        # hm_sum = torch.sum(heatmap, dim=(1, 2)) # B
        # hm_zero_indices = hm_sum < 1e-8
        
        # # for numerical stability
        # if sum(hm_zero_indices) > 0:
        #     uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
        #     heatmap[hm_zero_indices] += uniform_w[:, None, None]
        #     heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
        #     hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        # point = torch.sum(grid * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (B, 2)

        # return point

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

        points = torch.sum(grid.float() * heatmap.unsqueeze(-1).float(), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (F, 2)
        
        return points

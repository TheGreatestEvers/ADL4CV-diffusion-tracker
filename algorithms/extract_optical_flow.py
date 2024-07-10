import os
import pickle
import torch
from copy import deepcopy
from torchvision.models.optical_flow import raft_small
import torchvision.transforms.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_video(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    return batch

@torch.no_grad()
def extract_optical_flow_pairs(video: torch.Tensor, query_points: torch.Tensor):
    """
        video: FxCxHxW
        query_points: Nx3
    """

    F, C, H, W = video.shape

    model = raft_small(pretrained=True, progress=False).to(device).eval()

    reversed_video = video.flip(dims=[0])

    forward_flow = model(video[:-1], video[1:])[-1] #N, 2, H, W
    backward_flow = model(reversed_video[:-1], reversed_video[1:])[-1]

    query_points = query_points.cpu()
    forward_flow = forward_flow.cpu()
    backward_flow = backward_flow.cpu()

    point_pairs = []

    for query_point in query_points:
        t_0 = query_point[0].long()
        for sequence_length in range(t_0+1, F-1):
            intermediate_point = deepcopy(query_point[1:])
            outside_of_bounds = False

            for i in range(sequence_length):
                intermediate_point += forward_flow[i, :, intermediate_point.long()[0], intermediate_point.long()[1]]

                if intermediate_point[0] < 0 or intermediate_point[1] < 0 or intermediate_point[0] > H-1 or intermediate_point[1] > W-1:
                    outside_of_bounds = True
                    break

            if outside_of_bounds:
                break

            endpoint = deepcopy(intermediate_point)

            for i in range(sequence_length):
                intermediate_point += backward_flow[F-2-i, :, intermediate_point.long()[0], intermediate_point.long()[1]]

                if intermediate_point[0] < 0 or intermediate_point[1] < 0 or intermediate_point[0] > H-1 or intermediate_point[1] > W-1:
                    outside_of_bounds = True
                    break
            
            #if not outside_of_bounds and torch.equal(intermediate_point.long(), query_point[1:].long()):
            if not outside_of_bounds and torch.norm(intermediate_point - query_point[1:]) < 2:
                endpoint = torch.cat([torch.tensor([sequence_length]), endpoint])
                point_pair = (query_point, endpoint)
                point_pairs.append(point_pair)

    return point_pairs


def plot_images_with_points(images, points):
    """
    Plots two images with points on them.
    
    Parameters:
    images (torch.Tensor): A tensor of shape (2, C, H, W) containing two images.
    points (list of tuples): A list of two tuples, each containing the coordinates of the point to be drawn on each image.
    """
    # Check if the input tensor has the correct shape
    if images.shape[0] != 2 or len(images.shape) != 4:
        raise ValueError("The input tensor should have shape (2, C, H, W).")

    fig, axes = plt.subplots(1, 2)
    
    for i, ax in enumerate(axes):
        # Convert the image tensor to numpy for plotting
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Plot the image
        ax.imshow(img)
        
        # Get the point coordinates
        x, y = points[i]
        
        # Draw the point on the image
        ax.plot(x, y, 'ro')  # 'ro' means red color, circle marker
        
        # Optionally, set titles for each subplot
        ax.set_title(f'Image {i + 1}')
    
    plt.show()



if __name__ == "__main__":
    with open('/home/max/ADL4CV/features/davis/video_0.pkl', 'rb') as video_file:
        video = pickle.load(video_file)

    video = torch.tensor(video['video'][0], device=device).permute(0, 3, 1, 2)[:15]
    video = preprocess_video(video)

    cont_loop = True

    query_points = torch.rand([10, 3]) * 255
    query_points[:,0] = torch.rand([10]) * 16
    point_pairs = extract_optical_flow_pairs(video, query_points)

    print(point_pairs)

    for point_pair in point_pairs:
        images = torch.cat((video[point_pair[0][0].long()].unsqueeze(0), video[point_pair[1][0].long()].unsqueeze(0)), dim=0)
        print(point_pair[1][0])
        point_pair = [(point_pair[0][1], point_pair[0][2]), (point_pair[1][1], point_pair[1][2])]
    
        plot_images_with_points(images, point_pair)

    query_points = []
    endpoints = []

    for point_pair in point_pairs:
        query_points.append(point_pair[0])
        endpoints.append(point_pair[1])

    query_points = torch.stack(query_points)
    endpoints = torch.stack(endpoints)
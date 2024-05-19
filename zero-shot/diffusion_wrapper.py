import os
import gc
import torch
import torchvision.transforms.functional as functional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet3DConditionModel, DDIMScheduler
from diffusers import DiffusionPipeline

from utils import load_video

class DiffusionWrapper:
    def __init__(
            self,
            model_path: str = './text-to-video-ms-1.7b'
    ):
        
        self.model_path = model_path if os.path.exists(model_path) else 'damo-vilab/text-to-video-ms-1.7b'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16, variant="fp16")
        self.pipeline.enable_sequential_cpu_offload()
        self.pipeline.enable_vae_slicing()

        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

        #self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae", use_safetensor=True)
        #self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        #self.text_encoder = CLIPTextModel.from_pretrained(self.model_path, subfolder="text_encoder", use_safetensors=True)
        #self.unet = UNet3DConditionModel.from_pretrained(self.model_path, subfolder="unet", use_safetensor=True)
        #self.schedular = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")

        #self.vae = self.vae.to(self.device)
        #self.text_encoder = self.text_encoder.to(self.device)
        #self.unet = self.unet.to(self.device)

        self.feature_maps = {'up_block': [], 'down_block': [], 'mid_block': []}
        self.hooks = []

        def hook_feat_map_up(mod, inp, out):
            self.feature_maps['up_block'].append(out.cpu())
        def hook_feat_map_down(mod, inp, out):
            self.feature_maps['down_block'].append(out[0].cpu())
        def hook_feat_map_mid(mod, inp, out):
            self.feature_maps['mid_block'].append(out.cpu())

        for up_block in self.unet.up_blocks:
            self.hooks.append(up_block.register_forward_hook(hook_feat_map_up))
        for down_block in self.unet.down_blocks:
            self.hooks.append(down_block.register_forward_hook(hook_feat_map_down))
        self.hooks.append(self.unet.mid_block.register_forward_hook(hook_feat_map_mid))

    def concatenate_video_features(self, features):
        """
        Concatenates video feature tensors after resizing them to a uniform size.

        Args:
            features (dict): A dictionary containing feature maps. The dictionary keys can be considered as feature names
                            and the values are lists of feature tensors.

        Returns:
            torch.Tensor: A single concatenated feature map tensor of shape (BxFxCfxHxW)
        """

        max_height_width = max(ft.shape[-1] for fts in features.values() for ft in fts)

        feature_map = torch.cat([functional.resize(ft, [max_height_width] * 2) for fts in features.values() for ft in fts], dim=1)
        
        return feature_map



    def extract_video_features(
            self,
            video: torch.Tensor | str,
            prompt: str = ""
    ):
        """
        Extract video diffusion features from a video tensor or a video file.

        Args:
            video: A video input which can be either:
                - A torch.Tensor of shape (BxFxHxWxC)
                - A string path to a video file.
            prompt: A caption or text prompt for text-to-video diffusion.

        Returns:
            feature_map: A dictionary of feature map tensors, where each key corresponds to a different level of extracted features.
        """

        with torch.no_grad():

            if isinstance(video, str):
                video = load_video(video).unsqueeze(0)

            video = video.permute(0, 1, 4, 2, 3)

            B, F, C, H, W = video.shape

            video = video.to(dtype=torch.float16 , device=self.device) / 255.0

            tokenized_prompt = self.tokenizer(
                [prompt]*B, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )

            text_embeddings = self.text_encoder(tokenized_prompt.input_ids.to(self.device))[0]


            video = video.view(B*F, C, H, W)

            latent_video = self.vae.encode(video).latent_dist.sample()

            _, LC, LH, LW = latent_video.shape
            latent_video = latent_video.view(B, F, LC, LH, LW)

            noise = torch.randn(latent_video.shape, dtype=torch.float16, device=self.device)
            latent_video = self.scheduler.add_noise(latent_video, noise, torch.tensor(1, device=self.device))

            for key in self.feature_maps.keys():
                self.feature_maps[key].clear()

            self.unet(latent_video.permute(0, 2, 1, 3, 4), torch.tensor(1, device=self.device), encoder_hidden_states=text_embeddings).sample


        torch.cuda.empty_cache()
        gc.collect()

        return self.feature_maps


if __name__ == '__main__':
    diffusion_wrapper = DiffusionWrapper()
    feature_dict = diffusion_wrapper.extract_video_features('rocket256.gif', 'A rocket starting on Mars')
    feature_map = diffusion_wrapper.concatenate_video_features(feature_dict)

    print(feature_map.shape)
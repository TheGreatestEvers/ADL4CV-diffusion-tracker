import os
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
        self.pipeline.enable_model_cpu_offload()

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

        self.feature_maps = []
        self.hooks = []

        def hook_feat_map(mod, inp, out):
            self.feature_maps.append(out)

        for up_block in self.unet.up_blocks:
            self.hooks.append(up_block.register_forward_hook(hook_feat_map))


    def extract_video_features(
            self,
            video: torch.Tensor | str,
            prompt: str = ""
    ):
        """
        Extract video diffusion features from a heatmap.

        Args:
            video: torch.Tensor (BxFxHxWxC) | str path to video file
            prompt: caption for text-to-video diffusion

        Returns:
            feature map tensor (BxFxC_fxH/8xW/8)
        """

        if isinstance(video, str):
            video = load_video(video).unsqueeze(0)

        video = video.permute(0, 1, 4, 2, 3)

        B, F, C, H, W = video.shape

        video = video.to(dtype=torch.float16 , device=self.device) / 255.0

        tokenized_prompt = self.tokenizer(
            [prompt]*B, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokenized_prompt.input_ids.to(self.device))[0]


        video = video.view(B*F, C, H, W)
        with torch.no_grad():
            latent_video = self.vae.encode(video).latent_dist.sample()

        _, LC, LH, LW = latent_video.shape
        latent_video = latent_video.view(B, F, LC, LH, LW)

        noise = torch.randn(latent_video.shape, dtype=torch.float16, device=self.device)
        latent_video = self.scheduler.add_noise(latent_video, noise, torch.tensor(1, device=self.device))

        self.feature_maps.clear()

        with torch.no_grad():
            self.unet(latent_video.permute(0, 2, 1, 3, 4), torch.tensor(1, device=self.device), encoder_hidden_states=text_embeddings).sample

        max_height_width = 0
        for feature_map in self.feature_maps:
            max_height_width = feature_map.shape[-1] if max_height_width < feature_map.shape[-1] else max_height_width

        adj_feature_map = torch.cat([functional.resize(feature_map, [max_height_width]*2) for feature_map in self.feature_maps], dim=1)
        _, FC, FH, FW = adj_feature_map.shape
        adj_feature_map = adj_feature_map.view(B, F, FC, FH, FW)

        return adj_feature_map


if __name__ == '__main__':
    diffusion_wrapper = DiffusionWrapper()
    feature_map = diffusion_wrapper.extract_video_features('rocket256.gif', 'A rocket starting on Mars')
    print(feature_map.shape)
        
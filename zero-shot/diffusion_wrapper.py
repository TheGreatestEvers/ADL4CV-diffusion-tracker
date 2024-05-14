import os
import torch
import torchvision.transforms.functional as F
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
        if not isinstance(video, torch.Tensor):
            video = load_video(video)

        video = video.to(dtype=torch.float16 , device=self.device) / 255.0

        
        tokenized_prompt = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokenized_prompt.input_ids.to(self.device))[0]


        with torch.no_grad():
            latent_video = self.vae.encode(video).latent_dist.sample()


        noise = torch.randn(latent_video.shape, dtype=torch.float16, device=self.device)
        latent_video = self.scheduler.add_noise(latent_video, noise, torch.tensor(1, device=self.device))

        self.feature_maps.clear()

        with torch.no_grad():
            noise_pred = self.unet(latent_video.unsqueeze(0).permute(0, 2, 1, 3, 4), torch.tensor(1, device=self.device), encoder_hidden_states=text_embeddings).sample

        max_height_width = 0
        for feature_map in self.feature_maps:
            max_height_width = feature_map.shape[-1] if max_height_width < feature_map.shape[-1] else max_height_width

        adj_feature_map = torch.cat([F.resize(feature_map, [max_height_width]*2) for feature_map in self.feature_maps], dim=1)

        return adj_feature_map


if __name__ == '__main__':
    diffusion_wrapper = DiffusionWrapper()
    feature_map = diffusion_wrapper.extract_video_features('rocket256.gif', 'A rocket starting on Mars')
    print(feature_map.shape)
        
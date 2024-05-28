import os
import gc
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet3DConditionModel, DDIMScheduler
from diffusers import DiffusionPipeline

from algorithms.utils import load_video

class DiffusionWrapper:
    def __init__(
            self,
            model_path: str = './text-to-video-ms-1.7b',
            use_decoder_features: bool = True,
            enable_vae_slicing: bool = True
    ):
        
        self.model_path = model_path if os.path.exists(model_path) else 'damo-vilab/text-to-video-ms-1.7b'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pipeline = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16, variant="fp16")
        self.pipeline.enable_sequential_cpu_offload()
        if enable_vae_slicing:
            self.pipeline.enable_vae_slicing()

        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

        self.num_inference_steps = 25
        self.scaling_factor = 0.18215

        self.use_decoder_features = use_decoder_features

        #self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae", use_safetensor=True)
        #self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        #self.text_encoder = CLIPTextModel.from_pretrained(self.model_path, subfolder="text_encoder", use_safetensors=True)
        #self.unet = UNet3DConditionModel.from_pretrained(self.model_path, subfolder="unet", use_safetensor=True)
        #self.schedular = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")

        #self.vae = self.vae.to(self.device)
        #self.text_encoder = self.text_encoder.to(self.device)
        #self.unet = self.unet.to(self.device)

        self.feature_maps = {'up_block': [], 'down_block': [], 'mid_block': [], 'decoder_block': []}
        self.hooks = []

        def hook_feat_map_up(mod, inp, out):
            self.feature_maps['up_block'].append(out.cpu())
        def hook_feat_map_down(mod, inp, out):
            self.feature_maps['down_block'].append(out[0].cpu())
        def hook_feat_map_mid(mod, inp, out):
            self.feature_maps['mid_block'].append(out.cpu())
        def hook_feat_map_decoder(mod, inp, out):
            decoder_feature_maps = self.feature_maps['decoder_block']

            if out.shape[1:-1] not in [decoder_feature_map.shape[1:-1] for decoder_feature_map in decoder_feature_maps]:
                decoder_feature_maps.append(out.cpu())
            else:
                for idx, decoder_feature_map in enumerate(decoder_feature_maps):
                    if decoder_feature_map.shape[1:-1] == out.shape[1:-1]:
                        decoder_feature_maps[idx] = torch.cat((decoder_feature_map, out.cpu()), dim=0)

        for up_block in self.unet.up_blocks:
            self.hooks.append(up_block.register_forward_hook(hook_feat_map_up))
        for down_block in self.unet.down_blocks:
            self.hooks.append(down_block.register_forward_hook(hook_feat_map_down))
        self.hooks.append(self.unet.mid_block.register_forward_hook(hook_feat_map_mid))
        for decoder_block in self.vae.decoder.up_blocks:
            self.hooks.append(decoder_block.register_forward_hook(hook_feat_map_decoder))

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
            video = (video - 0.5) * 2

            tokenized_prompt = self.tokenizer(
                [prompt]*B, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )

            text_embeddings = self.text_encoder(tokenized_prompt.input_ids.to(self.device))[0]

            video = video.view(B*F, C, H, W)

            latent_video = self.vae.encode(video).latent_dist.sample()
            latent_video = latent_video * self.scaling_factor

            _, LC, LH, LW = latent_video.shape
            latent_video = latent_video.view(B, F, LC, LH, LW)
            latent_video = latent_video.permute(0, 2, 1, 3, 4) #B, LC, F, LH, LW

            self.scheduler.set_timesteps(self.num_inference_steps)
            last_scheduler_step = self.scheduler.timesteps[-1]

            noise = torch.randn(latent_video.shape, dtype=torch.float16, device=self.device)
            latent_video = self.scheduler.add_noise(latent_video, noise, last_scheduler_step)

            for key in self.feature_maps.keys():
                self.feature_maps[key].clear()

            noise_pred = self.unet(latent_video, last_scheduler_step, encoder_hidden_states=text_embeddings).sample

            if self.use_decoder_features:
                latent_video = self.scheduler.step(noise_pred, last_scheduler_step, latent_video).prev_sample
                latent_video = latent_video.permute(0, 2, 1, 3, 4) #B, F, LC, LH, LW
                latent_video = latent_video.view(B*F, LC, LH, LW)

                latent_video = 1 / self.scaling_factor * latent_video

                self.vae.decode(latent_video).sample

        torch.cuda.empty_cache()
        gc.collect()

        return self.feature_maps

if __name__ == '__main__':
    diffusion_wrapper = DiffusionWrapper()
    feature_dict = diffusion_wrapper.extract_video_features('rocket256.gif', 'A rocket starting on Mars')
    feature_map = diffusion_wrapper.concatenate_video_features(feature_dict)

    print(feature_map.shape)
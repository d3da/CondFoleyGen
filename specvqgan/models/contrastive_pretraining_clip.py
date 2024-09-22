import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchaudio

# import transformers
import languagebind as lb

sys.path.insert(0, '.')

from train import instantiate_from_config


class ContrastivePretraining(pl.LightningModule):
    def __init__(self,
                 video_encoder_config,
                 audio_encoder_config,
                 label_encoder_config,
                 video_key='video_path',
                 audio_key='audio_path',
                 label_key='label',
                 start_time_key='start_time',
                 end_time_key='end_time',
                 hit_class_key='hit_class'):

        super().__init__()

        self.video_encoder = instantiate_from_config(video_encoder_config)
        self.audio_encoder = instantiate_from_config(audio_encoder_config)
        self.label_encoder = instantiate_from_config(label_encoder_config)

        self.video_key = video_key
        self.audio_key = audio_key
        self.label_key = label_key
        self.start_time_key = start_time_key
        self.end_time_key = end_time_key
        self.hit_class_key = hit_class_key

    def forward(self, video, audio, label, start_times, end_times):
        emb_v = self.video_encoder(video, start_times, end_times)
        emb_a = self.audio_encoder(audio, start_times, end_times)
        with torch.no_grad():
            emb_l = self.label_encoder(label)
        return emb_v, emb_a, emb_l

    def shared_step(self, batch):
        emb_v, emb_a, emb_l = self(batch[self.video_key],
                                   batch[self.audio_key],
                                   batch[self.label_key],
                                   batch[self.start_time_key],
                                   batch[self.end_time_key])

        # TODO change loss
        loss = F.mse_loss(emb_v, emb_l) + F.mse_loss(emb_a, emb_l)
        print(loss)
        return loss

    def training_step(self, batch, *args, **kwargs):
        return self.shared_step(batch)

    def configure_optimizers(self):
        pass





class LB_VideoEncoder(pl.LightningModule):
    """
    TODO: This model also contains the CLIP text encoder, which is loaded multiple times
    May take up a lot more GPU space
    """

    def __init__(self,
                 pretrained_ckpt: str = 'LanguageBind/LanguageBind_Video_FT',
                 cache_dir: str = './cache_dir'):
        super().__init__()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=cache_dir).to(device=self.device)

        self.model.config.vision_config.video_decode_backend = 'pytorchvideo'

        self.modality_transform = lb.LanguageBindVideoProcessor(self.model.config)

    def forward(self, input_paths, clip_start_times, clip_end_times):
        input = self.process_input_video(input_paths, clip_start_times, clip_end_times)
        output = self.model.vision_model(pixel_values=input)[1]
        output_projected = self.model.visual_projection(output)
        return output_projected

    def process_input_video(self, input_paths, clip_start_times, clip_end_times):
        """
        We do this instead of using VideoProcessor.__call__ so we can supply the start and end times
        """
        processor = self.modality_transform

        pixel_values = [processor.image_processor(input_path,
                        processor.transform,
                        video_decode_backend='pytorchvideo',
                        clip_start_sec=clip_start_time,
                        clip_end_sec=clip_end_time,
                        num_frames=None)['video']
        for input_path, clip_start_time, clip_end_time
        in zip(input_paths, clip_start_times, clip_end_times)]

        return torch.stack(pixel_values).to(device=self.device)


class LB_AudioEncoder(pl.LightningModule):
    """
    TODO: This model also contains the CLIP text encoder, which is loaded multiple times
    May take up a lot more GPU space
    """

    def __init__(self,
                 pretrained_ckpt: str = 'LanguageBind/LanguageBind_Audio_FT',
                 cache_dir: str = './cache_dir'):
        super().__init__()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
        self.modality_transform = lb.LanguageBindAudioProcessor(self.model.config)

    def forward(self, input_paths, start_times, end_times):
        input = self.process_input_audio(input_paths, start_times, end_times)
        output = self.model.vision_model(pixel_values=input)[1]
        output_projected = self.model.visual_projection(output)
        return output_projected


    def process_input_audio(self, input_paths, start_times, end_times):
        processor = self.modality_transform

        pixel_values = []
        for input_path, start_time, end_time in zip(input_paths, start_times, end_times):
            waveform, sample_rate = lb.audio.processing_audio.torchaudio_loader(input_path)
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            waveform = waveform[:, start_frame : end_frame]

            pixel_values.append(processor.transform((waveform, sample_rate)))

        return torch.stack(pixel_values).to(device=self.device)


class LB_LabelEncoder(pl.LightningModule):
    """
    TODO: This model also contains the CLIP video encoder, which is loaded multiple times
    May take up a lot more GPU space
    """

    def __init__(self,
                 pretrained_ckpt: str = 'LanguageBind/LanguageBind_Video_FT',
                 cache_dir: str = './cache_dir'):
        super().__init__()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
        self.tokenizer = lb.LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)

    def forward(self, text):
        tokens_mask = self.tokenizer(text,
                                     max_length=77,
                                     padding='max_length',
                                     return_tensors='pt',
                                     truncation=True).to(device=self.device)

        output = self.model.text_model(**tokens_mask)[1]
        output_projected = self.model.text_projection(output)

        return output_projected

    

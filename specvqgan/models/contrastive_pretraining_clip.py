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
                 end_time_key='end_time'):

        super().__init__()

        self.video_encoder = instantiate_from_config(video_encoder_config)
        self.audio_encoder = instantiate_from_config(audio_encoder_config)
        self.label_encoder = instantiate_from_config(label_encoder_config)

        self.video_key = video_key
        self.audio_key = audio_key
        self.label_key = label_key
        self.start_time_key = start_time_key
        self.end_time_key = end_time_key

    def forward(self, video, audio, label, start_times, end_times):
        emb_v = self.video_encoder(video, start_times, end_times)
        emb_a = self.audio_encoder(audio)
        with torch.no_grad():
            emb_l = self.label_encoder(label)
        return emb_v, emb_a, emb_l

    def shared_step(self, batch):
        print(batch)
        # v = self.get_input(self.video_key, batch)
        # a = self.get_input(self.audio_key, batch)
        # l = self.get_input(self.label_key, batch)
        # start_times = self.get_input(self.start_time_key, batch)
        # end_times = self.get_input(self.end_time_key, batch)
        # emb_v, emb_a, emb_l = self(v, a, l, start_times, end_times)
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
        self.model = lb.LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)

        self.model.config.vision_config.video_decode_backend = 'pytorchvideo'

        self.modality_transform = lb.LanguageBindVideoProcessor(self.model.config)
        print(self.model.config)

    def forward(self, input_paths, clip_start_times, clip_end_times):
        input = self.process_input_video(input_paths, clip_start_times, clip_end_times)
        # print(input)
        # import pdb; pdb.set_trace()
        output = self.model.vision_model(pixel_values=input)[1]
        output_projected = self.model.visual_projection(output)
        # TODO Normalize the projected output? (https://github.com/PKU-YuanGroup/LanguageBind/blob/7070c53375661cdb235801176b564b45f96f0648/languagebind/__init__.py#L80)
        return output_projected

    def process_input_video(self, input_paths, clip_start_times, clip_end_times):
        """
        We do this instead of using VideoProcessor.__call__ so we can supply the start and end times
        """
        processor = self.modality_transform
        # for input_path, clip_start_time, clip_end_time in zip(input_paths, clip_start_times, clip_end_times):
        #     print(input_path)
        #     print(clip_start_time.item())
        #     print(clip_end_time.item())
        #     x = processor.image_processor(input_path,
        #                                   processor.transform,
        #                                   video_decode_backend='pytorchvideo',
        #                                   clip_start_sec=clip_start_time.item(),
        #                                   clip_end_sec=clip_end_time.item(),
        #                                   num_frames=None)
        #     print(x)
        #     import pdb; pdb.set_trace()

        return torch.stack([processor.image_processor(input_path,
                                                      processor.transform,
                                                      video_decode_backend='pytorchvideo',
                                                      clip_start_sec=clip_start_time,
                                                      clip_end_sec=clip_end_time,
                                                      num_frames=None)['video']
                            for input_path, clip_start_time, clip_end_time
                            in zip(input_paths, clip_start_times, clip_end_times)]).to(input_paths)



    
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

    def forward(self, input_paths):
        input = self.modality_transform(images=input_paths)
        output = self.model.vision_model(**input)[1]
        output_projected = self.model.visual_projection(output)
        # TODO Normalize the projected output? (https://github.com/PKU-YuanGroup/LanguageBind/blob/7070c53375661cdb235801176b564b45f96f0648/languagebind/__init__.py#L80)
        return output_projected
    

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
                                     truncation=True)

        output = self.model.text_model(**tokens_mask)[1]
        output_projected = self.model.text_projection(output)

        return output_projected

    

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
                 loss_config,
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
        self.loss_fn = instantiate_from_config(loss_config)

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
        classes = batch[self.hit_class_key]
        loss = self.loss_fn(emb_v, emb_a, emb_l, classes)

        return loss


    def training_step(self, batch, *args, **kwargs):
        return self.shared_step(batch)

    def validation_step(self, batch, *args, **kwargs):
        return self.shared_step(batch)


class ContrastiveLoss(pl.LightningModule):
    """
    Base class for contrastive loss function
    """

    def __init__(self):
        super().__init__()

    def forward(self, emb_v, emb_a, emb_l, classes):
        raise NotImplementedError

    def vector_distance_matrix(self, emb1, emb2):
        """
        Calculate the vector distance between all pairs of vectors between two matrices.
        Resulting m_dist[i][j] contains the euclidean distance between emb1[i] and emb2[j]
        
        Input sizes:
            - emb1 ~ (batch_size, vector_size)
            - emb2 ~ (batch_size, vector_size)
        Output sizes:
            - m_dist ~ (batch_size, batch_size)
        """
        assert emb1.shape == emb2.shape
        batch_size = emb1.shape[0]

        e1 = emb1.expand(batch_size, batch_size, -1)
        e2 = emb2.expand(batch_size, batch_size, -1)
        e1 = e1.permute(1, 0, 2)

        m_dist = (e1 - e2).square().sum(dim=2).sqrt()

        return m_dist


class BindToLabelEmbeddingContrastiveLoss(ContrastiveLoss):
    """
    Contrastive loss where the video/audio embeddings are pushed towards
    the label embeddings of the hit class in the current batch
    and pushed away from others (up to distance epsilon)
    """

    def __init__(self, epsilon):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, emb_v, emb_a, emb_l, classes):
        m_sim, m_mask = self.similarity_matrices(classes)

        v_dist = self.vector_distance_matrix(emb_v, emb_l)  # B x B
        a_dist = self.vector_distance_matrix(emb_a, emb_l)  # B x B
        zeros = torch.zeros_like(m_mask)

        v_loss_positive = v_dist * m_sim * m_mask
        v_loss_negative = (self.epsilon - v_dist) * ~m_sim * m_mask
        v_loss_negative = v_loss_negative.maximum(zeros)
        v_loss = v_loss_positive + v_loss_negative

        a_loss_positive = a_dist * m_sim * m_mask
        a_loss_negative = (self.epsilon - a_dist) * ~m_sim * m_mask
        a_loss_negative = a_loss_negative.maximum(zeros)
        a_loss = a_loss_positive + a_loss_negative

        # b = emb_v.shape[0]
        # for i in range(0, b):
        #     for j in range(0, b):
        #         matrix_loss = v_loss[i][j]
        #         d = (emb_v[i] - emb_l[j]).square().sum().sqrt()
        #         if classes[i] == classes[j]:
        #             l = d
        #         elif i > j:
        #             l = 0
        #         else:
        #             l = max(0, self.epsilon - d)
                # print('Matrix:', matrix_loss, 'Manual:', l)

        return v_loss + a_loss


    def similarity_matrices(self, classes):
        batch_size = classes.shape[0]
        
        c1 = classes.expand(batch_size, batch_size)
        c2 = classes.expand(batch_size, batch_size)
        c1 = c1.T

        m_sim = c1.eq(c2)
        # m_sim[i][j] contains (classes[i] == classes[j])
        m_mask = torch.ones_like(m_sim).tril().T
        # m_mask contains ones on and above the diagonal, zeros below the diagonal

        return m_sim, m_mask



class BindVideoAudioLoss(ContrastiveLoss):
    """
    Bind video and audio together directly, without using the labels.
    Video embedding is pushed towards the audio embedding from the same sample,
    pushed away from other audio embeddings in the batch, up to distance epsilon.
    Vice versa for audio
    """

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, emb_v, emb_a, emb_l, classes):
        batch_size = emb_v.shape[0]
        m_sim, m_mask = self.similarity_matrices(batch_size)

        m_dist = self.vector_distance_matrix(emb_v, emb_a)
        zeros = torch.zeros_like(m_mask)

        loss_positive = m_dist * m_sim * m_mask
        loss_negative = (self.epsilon - m_dist) * ~m_sim * m_mask
        loss_negative = loss_negative.maximum(zeros)
        loss = (loss_positive + loss_negative)

        return loss

    def similarity_matrices(self, batch_size):
        m_sim = torch.eye(batch_size, dtype=torch.bool)
        # m_sim contains true on the diagonal, false elsewhere
        m_mask = torch.ones([batch_size, batch_size]).tril().T
        # m_mask contains ones on and above the diagonal, zeros under the diagonal

        return m_sim.to(device=self.device), m_mask.to(device=self.device)


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

    

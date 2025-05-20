import torch
import pytorch_lightning as pl
import languagebind as lb
from itertools import chain

import sys
sys.path.insert(0, '..')

from utils import instantiate_from_config

class V2EncoderTraining(pl.LightningModule):
    def __init__(self,
                 a_encoder_config,
                 v_encoder_config,
                 semantic_loss_config,
                 temporal_loss_config,
                 label_embeddings_path,
                 optim_learn_rate,
                 optim_weight_decay):
        super().__init__()
        self.a_encoder = instantiate_from_config(a_encoder_config)
        self.v_encoder = instantiate_from_config(v_encoder_config)
        self.semantic_loss = instantiate_from_config(semantic_loss_config)
        self.temporal_loss = instantiate_from_config(temporal_loss_config)

        label_emb_dict = torch.load(label_embeddings_path)
        self.register_buffer('label_embeddings',
                             label_emb_dict['label_embeddings'].requires_grad_(False))

        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay
        self.save_hyperparameters()

    def configure_optimizers(self):
        m = self.trainer.model
        params = (p for p in chain(m.a_encoder.parameters(),
                                   m.v_encoder.parameters())
                  if p.requires_grad)
        optimizer = torch.optim.Adam(params,
                                     lr=self.optim_learn_rate,
                                     weight_decay=self.optim_weight_decay)
        return optimizer


    def shared_step(self, batch, log_prefix):
        batch_size, num_segments, *video_shape = batch['video_data'].shape
        assert batch_size == batch['audio_data'].shape[0]
        assert num_segments == batch['audio_data'].shape[1]
        batch_size, num_segments, *audio_shape = batch['audio_data'].shape

        audio_data_reshaped = batch['audio_data'].reshape(batch_size * num_segments, *audio_shape)
        video_data_reshaped = batch['video_data'].reshape(batch_size * num_segments, *video_shape)
        hit_class_nums_reshaped = batch['hit_class_nums'].reshape(batch_size * num_segments)

        all_audio_embeddings = self.a_encoder(audio_data_reshaped)
        all_video_embeddings = self.v_encoder(video_data_reshaped)

        audio_embeddings_per_clip = all_audio_embeddings.reshape(batch_size, num_segments, -1)
        video_embeddings_per_clip = all_video_embeddings.reshape(batch_size, num_segments, -1)

        a_semantic_loss = self.semantic_loss(all_audio_embeddings,
                                           hit_class_nums_reshaped,
                                           self.label_embeddings)
        self.log(f'{log_prefix}/audio_semantic_loss', a_semantic_loss, on_step=True, prog_bar=True, batch_size=batch_size)

        v_semantic_loss = self.semantic_loss(all_video_embeddings,
                                             hit_class_nums_reshaped,
                                             self.label_embeddings)
        self.log(f'{log_prefix}/video_semantic_loss', a_semantic_loss, on_step=True, prog_bar=True, batch_size=batch_size)

        temporal_loss = self.temporal_loss(audio_embeddings_per_clip, video_embeddings_per_clip)
        self.log(f'{log_prefix}/temporal_loss', temporal_loss, on_step=True, prog_bar=True, batch_size=batch_size)

        combined_loss = a_semantic_loss + v_semantic_loss + temporal_loss
        return combined_loss, batch_size

    def training_step(self, batch, batch_idx):
        loss, batch_size = self.shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss, batch_size = self.shared_step(batch, 'validation')
        self.log('hp_metric', loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss, batch_size = self.shared_step(batch, 'test')
        return loss

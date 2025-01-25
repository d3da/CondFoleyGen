import pytorch_lightning as pl
import torch
import itertools

from utils import instantiate_from_config

class TemporalAlignmentLearning(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 audio_align_model_config,
                 video_align_model_config,
                 alignment_loss_config,
                 optim_learn_rate,
                 optim_weight_decay,
                 ):
        super().__init__()

        self.encoder_model = instantiate_from_config(encoder_config)

        self.audio_align_model = instantiate_from_config(audio_align_model_config)
        self.video_align_model = instantiate_from_config(video_align_model_config)
        self.alignment_loss = instantiate_from_config(alignment_loss_config)

        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        m = self.trainer.model
        params = (p for p in itertools.chain(m.audio_align_model.parameters(),
                                             m.video_align_model.parameters())
                  if p.requires_grad)
        optimizer = torch.optim.Adam(params,
                                     lr=self.optim_learn_rate,
                                     weight_decay=self.optim_weight_decay)
        return {'optimizer': optimizer}

    def shared_step(self, batch, log_prefix):
        audio_embeddings, video_embeddings = [], []
        for clip_dict in batch:
            audio_emb, video_emb = self.encoder_model(clip_dict)
            if audio_emb is None or video_emb is None:
                continue
            audio_embeddings.append(audio_emb)
            video_embeddings.append(video_emb)

        if len(audio_embeddings) == 0 or len(video_embeddings) == 0:
            return None

        audio_embeddings = torch.stack(audio_embeddings)
        video_embeddings = torch.stack(video_embeddings)

        # audio_emb, video_emb = audio_emb.unsqueeze(0), video_emb.unsqueeze(0)
        aligned_audio_emb = self.audio_align_model(audio_embeddings)
        aligned_video_emb = self.video_align_model(video_embeddings)

        loss = self.alignment_loss(aligned_audio_emb, aligned_video_emb)
        self.log(f'{log_prefix}/loss', loss, prog_bar=True, on_step=True, batch_size=1)
        return loss

    def training_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'validation')
        self.log('hp_metric', loss, batch_size=1)
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'test')
        return loss



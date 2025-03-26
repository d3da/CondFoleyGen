import pytorch_lightning as pl
import torch
import itertools

from utils import instantiate_from_config

class TemporalAlignmentLearning(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 audio_tcc_model_config,
                 alignment_loss_config,
                 alignment_procedure_config,
                 optim_learn_rate,
                 optim_weight_decay,
                 ):
        super().__init__()

        self.encoder_model = instantiate_from_config(encoder_config)

        self.audio_tcc_model = instantiate_from_config(audio_tcc_model_config)
        self.alignment_loss = instantiate_from_config(alignment_loss_config)
        self.alignment_procedure = instantiate_from_config(alignment_procedure_config)

        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        m = self.trainer.model
        params = (p for p in m.audio_tcc_model.parameters()
                  if p.requires_grad)
        optimizer = torch.optim.Adam(params,
                                     lr=self.optim_learn_rate,
                                     weight_decay=self.optim_weight_decay)
        return {'optimizer': optimizer}

    def shared_step(self, batch, log_prefix):
        audio_embeddings, video_embeddings = [], []
        _shifted_video = []
        for clip_dict in batch:
            embeddings_dict = self.encoder_model(clip_dict)
            if embeddings_dict is None:
                continue
            audio_embeddings.append(embeddings_dict['shifted_audio'])
            video_embeddings.append(embeddings_dict['unshifted_video'])
            _shifted_video.append(embeddings_dict['shifted_video'])

        if len(audio_embeddings) == 0 or len(video_embeddings) == 0:
            return None

        audio_embeddings = torch.stack(audio_embeddings)
        video_embeddings = torch.stack(video_embeddings)
        _shifted_video = torch.stack(_shifted_video)

        tcc_audio_emb = self.audio_tcc_model(audio_embeddings)

        loss = self.alignment_loss(tcc_audio_emb, video_embeddings)
        self.log(f'{log_prefix}/loss', loss, prog_bar=True, on_step=True, batch_size=1)

        frobenius_norm = torch.linalg.matrix_norm(tcc_audio_emb)
        self.log(f'{log_prefix}/frobenius_norm', frobenius_norm.mean(), prog_bar=False, on_step=True, batch_size=1)

        a_re = tcc_audio_emb.reshape((-1, tcc_audio_emb.shape[-1]))
        v_re = _shifted_video.reshape((-1, tcc_audio_emb.shape[-1]))
        cos_sim_loss = torch.nn.functional.cosine_embedding_loss(a_re, v_re, torch.ones(a_re.shape[0]).to(device=self.device))
        self.log(f'{log_prefix}/cosine_loss', cos_sim_loss, prog_bar=True, on_step=True, batch_size=1)
        mse_loss = torch.nn.functional.mse_loss(tcc_audio_emb, _shifted_video)
        self.log(f'{log_prefix}/mse_loss', mse_loss, prog_bar=False, on_step=True, batch_size=1)
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



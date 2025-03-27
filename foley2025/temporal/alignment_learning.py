import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import itertools

from utils import instantiate_from_config

class TemporalAlignmentLearning(pl.LightningModule):
    def __init__(self,
                 encoder_config,
                 audio_tcc_model_config,
                 alignment_procedure_config,
                 tcc_loss_config,
                 alignment_loss_config,
                 tcc_loss_weight,
                 alignment_loss_weight,
                 optim_learn_rate,
                 optim_weight_decay,
                 ):
        super().__init__()

        self.encoder_model = instantiate_from_config(encoder_config)

        self.audio_tcc_model = instantiate_from_config(audio_tcc_model_config)
        self.alignment_procedure = instantiate_from_config(alignment_procedure_config)

        self.tcc_loss = instantiate_from_config(tcc_loss_config)
        self.alignment_loss = instantiate_from_config(alignment_loss_config)
        self.tcc_loss_weight = tcc_loss_weight
        self.alignment_loss_weight = alignment_loss_weight

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
        _unshifted_audio = []
        for clip_dict in batch:
            embeddings_dict = self.encoder_model(clip_dict)
            if embeddings_dict is None:
                continue
            audio_embeddings.append(embeddings_dict['shifted_audio'])
            video_embeddings.append(embeddings_dict['unshifted_video'])
            _shifted_video.append(embeddings_dict['shifted_video'])
            _unshifted_audio.append(embeddings_dict['unshifted_audio'])

        if len(audio_embeddings) == 0 or len(video_embeddings) == 0:
            return None

        batch_size = len(audio_embeddings)
        assert batch_size == len(video_embeddings) == len(_shifted_video) == len(_unshifted_audio)

        audio_embeddings = torch.stack(audio_embeddings)
        video_embeddings = torch.stack(video_embeddings)
        _shifted_video = torch.stack(_shifted_video)
        _unshifted_audio = torch.stack(_unshifted_audio)

        tcc_audio_emb = self.audio_tcc_model(audio_embeddings)

        # Calculate TCC loss (between tcc audio and video embeddings)
        tcc_loss = self.tcc_loss_weight * self.tcc_loss(tcc_audio_emb, video_embeddings)
        self.log(f'{log_prefix}/tcc_loss', tcc_loss, prog_bar=True, on_step=True, batch_size=batch_size)


        frobenius_norm = torch.linalg.matrix_norm(tcc_audio_emb)
        self.log(f'{log_prefix}/tcc_frobenius_norm', frobenius_norm.mean(), prog_bar=False, on_step=True, batch_size=batch_size)

        self.calculate_additional_metrics_tcc(log_prefix, batch_size, tcc_audio_emb, _shifted_video)

        # Compute explicit alignment of tcc embeddings to video
        aligned_audio_emb = self.alignment_procedure(tcc_audio_emb, video_embeddings)

        # Metrics calculated between aligned audio embedding and unshifted audio embeddings
        # align_mse_loss = F.mse_loss(aligned_audio_emb, _unshifted_audio)
        # self.log(f'{log_prefix}/aligned_mse_loss', align_mse_loss, prog_bar=True, on_step=True, batch_size=batch_size)

        # Calculate alignment loss (between aligned audio and unshifted audio)
        alignment_loss = self.alignment_loss_weight * self.alignment_loss(aligned_audio_emb, _unshifted_audio)
        self.log(f'{log_prefix}/aligned_loss', alignment_loss, prog_bar=True, on_step=True, batch_size=batch_size)

        combined_loss = tcc_loss + alignment_loss
        self.log(f'{log_prefix}/combined_loss', combined_loss, prog_bar=True, on_step=True, batch_size=batch_size)
        return combined_loss

    def calculate_additional_metrics_tcc(self,
                                         log_prefix,
                                         batch_size,
                                         tcc_audio_embeddings,
                                         shifted_video_embeddings):
        # Metrics calculated between (shifted) TCC audio embeddings and shifted video embeddings
        a_re = tcc_audio_embeddings.reshape((-1, tcc_audio_embeddings.shape[-1]))
        v_re = shifted_video_embeddings.reshape((-1, tcc_audio_embeddings.shape[-1]))
        cos_sim_loss = F.cosine_embedding_loss(a_re, v_re, torch.ones(a_re.shape[0]).to(device=self.device))
        self.log(f'{log_prefix}/cosine_loss', cos_sim_loss, prog_bar=False, on_step=True, batch_size=batch_size)
        mse_loss = F.mse_loss(tcc_audio_embeddings, shifted_video_embeddings)
        self.log(f'{log_prefix}/mse_loss', mse_loss, prog_bar=False, on_step=True, batch_size=batch_size)

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



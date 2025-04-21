import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from utils import instantiate_from_config


class DecoderTrainingAlignedAudio(pl.LightningModule):
    def __init__(self, *,
                 decoder_config,
                 encoder_config,
                 audio_tcc_model_config,
                 alignment_procedure_config,
                 optim_learn_rate,
                 optim_weight_decay):
        super().__init__()

        self.encoder_model = instantiate_from_config(encoder_config)
        self.decoder_model = instantiate_from_config(decoder_config)
        self.audio_tcc_model = instantiate_from_config(audio_tcc_model_config)
        self.audio_tcc_model.eval()

        self.alignment_procedure = instantiate_from_config(alignment_procedure_config)

        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay

        self.save_hyperparameters()

    def configure_optimizers(self):
        m = self.trainer.model
        params = (p for p in m.decoder_model.parameters()
                  if p.requires_grad)
        optimizer = torch.optim.Adam(params,
                                     lr=self.optim_learn_rate,
                                     weight_decay=self.optim_weight_decay)
        return {'optimizer': optimizer}

    def shared_step(self, batch, log_prefix, batch_idx):
        audio_embeddings = []
        video_embeddings = []
        unshifted_spectrogram = []
        for clip_dict in batch:
            embeddings_dict = self.encoder_model(clip_dict)
            if embeddings_dict is None:
                continue
            audio_embeddings.append(embeddings_dict['shifted_audio'])
            video_embeddings.append(embeddings_dict['unshifted_video'])
            unshifted_spectrogram.append(embeddings_dict['unshifted_spectrogram'])

        if len(audio_embeddings) == 0 or len(video_embeddings) == 0:
            return None

        batch_size = len(audio_embeddings)
        assert batch_size == len(video_embeddings) == len(unshifted_spectrogram)

        audio_embeddings = torch.stack(audio_embeddings)
        video_embeddings = torch.stack(video_embeddings)
        unshifted_spectrogram = torch.stack(unshifted_spectrogram)

        tcc_audio_emb = self.audio_tcc_model(audio_embeddings)
        aligned_audio_emb = self.alignment_procedure(tcc_audio_emb, video_embeddings)

        generated_spectrogram = self.decoder_model(aligned_audio_emb)
        loss = F.mse_loss(generated_spectrogram, unshifted_spectrogram)
        self.log(f'{log_prefix}/loss', loss, prog_bar=True, on_step=True, batch_size=batch_size)
        #TODO consider only the overlapping part of the image? Or maybe don't...
        #     if not, definitely calculate the overlapping-only loss as a metric
        if batch_idx % 50 == 0:
            self.log_image(unshifted_spectrogram, generated_spectrogram, log_prefix, batch_idx)

        return loss

    def log_image(self, unshifted_spectrogram, generated_spectrogram, log_prefix, step):
        if not self.logger.__class__.__name__ == 'WandbLogger':
            return
        self.logger.log_image(key=f'{log_prefix}/spec',
                              images=[unshifted_spectrogram[0].T, generated_spectrogram[0].T],
                              caption=['Ground-Truth', 'Predicted'],
                              step=self.trainer.global_step)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'train', batch_idx)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'validation', batch_idx)
        self.log('hp_metric', loss, batch_size=1)
        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'test', batch_idx)
        return loss
 

class DecoderTrainingStandardAudio(pl.LightningModule):
    def __init__(self, *, decoder_config, encoder_config, optim_learn_rate, optim_weight_decay):
        super().__init__()
        self.encoder_model = instantiate_from_config(encoder_config)
        self.decoder_model = instantiate_from_config(decoder_config)
        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay

        self.save_hyperparameters()

    def configure_optimizers(self):
        m = self.trainer.model
        params = (p for p in m.decoder_model.parameters()
                  if p.requires_grad)
        optimizer = torch.optim.Adam(params,
                                     lr=self.optim_learn_rate,
                                     weight_decay=self.optim_weight_decay)
        return {'optimizer': optimizer}

    def shared_step(self, batch, log_prefix, batch_idx):
        unshifted_audio_emb = []
        unshifted_spectrogram = []
        for clip_dict in batch:
            embeddings_dict = self.encoder_model(clip_dict)
            if embeddings_dict is None:
                continue
            unshifted_audio_emb.append(embeddings_dict['unshifted_audio'])
            unshifted_spectrogram.append(embeddings_dict['unshifted_spectrogram'])

        if len(unshifted_audio_emb) == 0:
            return None

        batch_size = len(unshifted_audio_emb)
        assert batch_size == len(unshifted_spectrogram)

        unshifted_audio_emb = torch.stack(unshifted_audio_emb)
        unshifted_spectrogram = torch.stack(unshifted_spectrogram)

        generated_spectrogram = self.decoder_model(unshifted_audio_emb)

        loss = F.mse_loss(generated_spectrogram, unshifted_spectrogram)
        self.log(f'{log_prefix}/loss', loss, prog_bar=True, on_step=True, batch_size=batch_size)

        if batch_idx % 50 == 0:
            self.log_image(unshifted_spectrogram, generated_spectrogram, log_prefix, batch_idx)

        return loss

    def log_image(self, unshifted_spectrogram, generated_spectrogram, log_prefix, step):
        if not self.logger.__class__.__name__ == 'WandbLogger':
            return
        self.logger.log_image(key=f'{log_prefix}/spec',
                              images=[unshifted_spectrogram[0].T, generated_spectrogram[0].T],
                              caption=['Ground-Truth', 'Predicted'],
                              step=self.trainer.global_step)
        

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'train', batch_idx)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'validation', batch_idx)
        self.log('hp_metric', loss, batch_size=1)
        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.shared_step(batch, 'test', batch_idx)
        return loss
 

"""
In av_cond_transformer.Net2NetTransformer we have the following

first_stage: VQGAN which takes audio and produces codebook indices
cond_stage: resnet r2plus1d_18 which takes video and produces indices
transformer: takes (zp,c) and predicts z
x: input audio (target) -> z: input audio codebook
xp: input conditional audio -> zp: conditional audio codebook
c: input video
"""
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchaudio

from omegaconf.listconfig import ListConfig

sys.path.insert(0, '.')
from train import instantiate_from_config
from specvqgan.data.transforms import Wave2Spectrogram, PitchShift, NormalizeAudio


SR = 22050


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class CLAVP(pl.LightningModule):
    """
    Contrastive Label Audio/Video Pairs

    """

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 a_embedder_config,
                 v_embedder_config,
                 l_embedder_config,
                 ckpt_path=None, ignore_keys=[],
                 first_stage_key='image',
                 cond_first_stage_key='cond_image',
                 cond_stage_key='feature',
                 label_key='label',
                 p_normalize=0.,
                 p_pitch_shift=0.,
                 mel_num=80,
                 spec_crop_len=160):
        super().__init__()
        self.init_first_stage_from_ckpt(first_stage_config)
        a_embed_dim = self.first_stage_model.embed_dim
        self.init_cond_stage_from_ckpt(cond_stage_config)

        self.wav_transforms = nn.Sequential(
            transforms.RandomApply([NormalizeAudio()], p=p_normalize),
            transforms.RandomApply([PitchShift()], p=p_pitch_shift),
            torchaudio.transforms.Spectrogram(
                n_fft=1024,
                hop_length=1024//4,
                power=1,
            ),
            torchaudio.transforms.MelScale(
                n_mels=80,
                sample_rate=SR,
                f_min=125,
                f_max=7600,
                n_stft=513,
                norm='slaney'
            ),
            Wave2Spectrogram(mel_num, spec_crop_len),
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        a_embedder_config.params.input_size = a_embed_dim
        print(a_embedder_config)
        self.audio_embedder = instantiate_from_config(config=a_embedder_config)
        self.video_embedder = instantiate_from_config(config=v_embedder_config)
        self.label_embedder = instantiate_from_config(config=l_embedder_config)

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.cond_first_stage_key = cond_first_stage_key
        self.label_key = label_key

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.cond_stage_model = model


    def forward(self, x, c, labels):
        """
        TODO change this for contrastive

        steps for contrastive:
        encode z (audio)
        encode c (video)
        embed z and c using audio and video embedders

        encode label
        return encoded z, c, (encoded label)
        """

        quant_z, _ = self.encode_to_z(x) # VQ-GAN encoding
        quant_c, _ = self.encode_to_c(c) # Conv1-1 down dim + col-major permuter
        label_emb = self.embed_label(labels)
        audio_emb = self.audio_embedder(quant_z)
        video_emb = self.video_embedder(quant_c)

        # print('Shape of quant_z, quant_c: ', end='')
        # print(quant_z.shape, end=' ')
        # print(quant_c.shape)
        # z_indices = z_indices[:, :self.clip]
        # a_indices = z_indices
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        # target = z_indices[:, self.clip:]
        # cz_indices = torch.cat((c_indices, a_indices), dim=1)
        # make the prediction
        # logits, _, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        # logits = logits[:, c_indices.shape[1]-1:]

        # return logits, target
        return audio_emb, video_emb, label_emb

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        # indices = self.first_stage_permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        # if self.downsample_cond_size > -1:
        #     c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, info = self.cond_stage_model.encode(c)
        indices = info[2].view(quant_c.shape[0], -1)
        # indices = self.cond_stage_permuter(indices)
        return quant_c, indices

    @torch.no_grad()
    def embed_label(self, labels):
        return self.label_embedder(labels)

    def get_input(self, key, batch):
        if isinstance(key, str):
            if key in ['feature', 'target']:
                x = self.cond_stage_model.get_input(batch, key, drop_cond=False)
            # if batch[key] is 1D; else the batch[key] is 2D
            else:
                x = batch[key]
                if hasattr(x, 'shape'):
                    if len(x.shape) == 3:
                        x = x[..., None]
                    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if hasattr(x, 'dtype') and x.dtype == torch.double:
                x = x.float()
        elif isinstance(key, ListConfig):
            x = self.cond_stage_model.get_input(batch, key)
            for k, v in x.items():
                if v.dtype == torch.double:
                    x[k] = v.float()
        return x


    def spec_transform(self, batch):
        wav = batch[self.first_stage_key]
        wav_cond = batch[self.cond_first_stage_key]
        N = wav.shape[0]
        wav_cat = torch.cat([wav, wav_cond], dim=0)
        self.wav_transforms.to(wav_cat.device)
        spec = self.wav_transforms(wav_cat.to(torch.float32))
        batch[self.first_stage_key] = 2 * spec[:N] - 1
        batch[self.cond_first_stage_key] = 2 * spec[N:] - 1
        return batch


    def get_xcl(self, batch, N=None):
        if len(batch[self.first_stage_key].shape) == 2:
            batch = self.spec_transform(batch)
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        lbl = self.get_input(self.label_key, batch)
        if N is not None:
            x = x[:N]
            if isinstance(self.cond_stage_key, ListConfig):
                c = {k: v[:N] for k, v in c.items()}
            else:
                c = c[:N]
        return x, c, lbl

    def shared_step(self, batch, batch_idx):
        x, c, lbl = self.get_xcl(batch)
        audio_emb, video_emb, label_emb = self(x, c, lbl)
        # TODO check shape
        # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        # TODO modify loss function to be better
        loss = F.mse_loss(audio_emb, label_emb) + F.mse_loss(video_emb, label_emb)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        """
        TODO set Adam hyperparameters
        """
        params = list(self.audio_embedder.ff.parameters()) + list(self.video_embedder.ff.parameters())
        print('params')
        print(params)
        return torch.optim.Adam(params)


    # def configure_optimizers(self):
    #     """
    #     Following minGPT:
    #     This long function is unfortunately doing something very simple and is being very defensive:
    #     We are separating out all parameters of the model into two buckets: those that will experience
    #     weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    #     We are then returning the PyTorch optimizer object.
    #     """
    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear, )

    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.LSTM, torch.nn.GRU)
    #     for mn, m in self.transformer.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)
    #             elif ('weight' in pn or 'bias' in pn) and isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add('pos_emb')

    #     # validate that we considered every parameter
    #     param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    #     assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    #                                                 % (str(param_dict.keys() - union_params), )

    #     # create the pytorch optimizer object
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]
    #     optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
    #     return optimizer


class LatentEmbedder(nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__()
        self.ff = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # linear layer requires the input_size as the last dimension
        x = self.ff(x)
        x = self.relu(x)
        return x


class AudioContrastiveLatentEmbedder(LatentEmbedder):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(input_size, output_size, *args, **kwargs)

    def forward(self, x):
        """
        x: batch_size, input_size, W, H
        """
        # print(x.shape)
        x = x.permute(0, 2, 3, 1)
        # import pdb; pdb.set_trace()
        return super().forward(x)


class VideoContrastiveLatentEmbedder(LatentEmbedder):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        print(x.shape)
        x = x.permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        return super().forward(x)


class LabelEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        # print(x.shape)
        return x


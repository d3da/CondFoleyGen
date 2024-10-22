import sys

import pytorch_lightning as pl
import torch

import languagebind as lb

sys.path.insert(0, '.')

from train import instantiate_from_config


class ContrastiveSingleModality(pl.LightningModule):
    """
    Trains a single modality encoder (audio or video) by binding the embeddings to precomputed label embeddings.
    Triplet loss is used
    """
    def __init__(self,
                 m_encoder_config,
                 loss_config,
                 m_key,
                 label_embeddings_path,
                 optim_learn_rate,
                 optim_weight_decay,
                 hit_class_key='hit_class'):
        super().__init__()
        self.save_hyperparameters()

        self.m_encoder = instantiate_from_config(m_encoder_config)
        self.loss_fn = instantiate_from_config(loss_config)

        self.m_key = m_key
        self.hit_class_key = hit_class_key

        label_emb_dict = torch.load(label_embeddings_path)

        # Use register_buffer so tensors will be moved to the correct device
        self.register_buffer('label_embeddings',
                             label_emb_dict['label_embeddings'].requires_grad_(False))
        self.register_buffer('label_none_mask',
                             label_emb_dict['none_mask'].requires_grad_(False))

        self.optim_learn_rate = optim_learn_rate
        self.optim_weight_decay = optim_weight_decay

    def configure_optimizers(self):
        # TODO set learn rate, etc
        m = self.trainer.model
        params = (p for p in m.m_encoder.parameters() if p.requires_grad)
        return torch.optim.Adam(params,
                                lr=self.optim_learn_rate,
                                weight_decay=self.optim_weight_decay)

    def forward(self, m_input):
        return self.m_encoder(m_input)

    def shared_step(self, batch, log_prefix):
        emb = self(batch[self.m_key])
        classes = batch[self.hit_class_key].to(device=self.device)

        loss, partial_loss_dict = self.loss_fn(emb,
                                               self.label_embeddings,
                                               classes,
                                               self.label_none_mask)

        self.log_dict({f'{log_prefix}/{k}': v for k, v in partial_loss_dict.items()},
                      prog_bar=False,
                      on_step=True,
                      batch_size=classes.shape[0])
        self.log(f'{log_prefix}/loss', loss, prog_bar=True, on_step=True, batch_size=classes.shape[0])

        return loss

    def training_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'validation')
        self.log('hp_metric', loss)
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'test')
        return loss


class SingleModalityTripletLoss(pl.LightningModule):
    """
    Given an embedding ~E, and a positive example E+ and negative example E-,
    the loss is defined as L = max(0, ||~E - E+|| + epsilon - ||~E - E-||)
    """
    def __init__(self, epsilon):
        super().__init__()

        self.epsilon = epsilon

    def choose_triplet(self, emb, label_emb, classes):
        """TODO: select 'hard' negative examples"""
        total_classes = label_emb.shape[0]
        distribution = []
        for n in range(emb.shape[0]):
            weights = torch.ones((total_classes), device=self.device)
            weights[classes[n]] = 0
            distribution.append(weights)

        negative_indices = torch.multinomial(torch.stack(distribution), 1).squeeze(dim=1)
        return label_emb[classes], label_emb[negative_indices]
        

    def forward(self, m_emb, pos_emb, neg_emb):

        pos_emb, neg_emb = self.choose_triplet(m_emb, label_emb, classes)

        assert m_emb.shape == pos_emb.shape == neg_emb.shape

        l_pos = (m_emb - pos_emb).square().sum(dim=1).sqrt()  # todo check dims
        l_neg = (m_emb - neg_emb).square().sum(dim=1).sqrt()

        zeros = torch.zeros_like(l_neg)
        loss = torch.max(zeros, l_pos + self.epsilon - l_neg)
        partial_loss_dict = dict(l_pos=l_pos.mean(),
                                 l_neg=-l_neg.mean())
        return loss.mean(), partial_loss_dict


class InfoNCELoss(pl.LightningModule):
    """
    Loss = - log \\frac{
    """

    def __init__(self, temperature=1., eps=1e-6, use_label_mask=False):
        super().__init__()
        self.temperature = temperature

        self.cosine_sim = torch.nn.CosineSimilarity(dim=2, eps=eps)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.use_label_mask = use_label_mask

    def forward(self, m_emb, label_emb, classes, label_mask):
        sim_matrix = self.cosine_sim(m_emb.unsqueeze(1),  # B x 1 x D
                                     label_emb.unsqueeze(0))  # 1 x L x D
        sim_matrix /= self.temperature  # B x L

        if self.use_label_mask:
            sim_matrix *= label_mask
            
        prob_logits = self.log_softmax(sim_matrix)

        num_classes = label_emb.shape[0]
        classes_one_hot = torch.nn.functional.one_hot(classes, num_classes=num_classes)  # B x L
        loss = -prob_logits.mul(classes_one_hot).sum(dim=1)  # B

        return loss.mean(), dict()



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
        self.save_hyperparameters()

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

    def configure_optimizers(self):
        # TODO set learn rate, etc.
        m = self.trainer.model
        params = list(m.video_encoder.parameters()) + list(m.audio_encoder.parameters())
        return torch.optim.Adam(params)

    def forward(self, video, audio, label, start_times, end_times):
        emb_v = self.video_encoder(video, start_times, end_times)
        emb_a = self.audio_encoder(audio, start_times, end_times)
        with torch.no_grad():
            emb_l = self.label_encoder(label)
        return emb_v, emb_a, emb_l

    def shared_step(self, batch, log_prefix):
        emb_v, emb_a, emb_l = self(batch[self.video_key],
                                   batch[self.audio_key],
                                   batch[self.label_key],
                                   batch[self.start_time_key],
                                   batch[self.end_time_key])
        classes = batch[self.hit_class_key]
        loss, partial_loss_dict = self.loss_fn(emb_v, emb_a, emb_l, classes)

        self.log_dict({f'{log_prefix}/{k}': v.sum() for k, v in partial_loss_dict.items()},
                      prog_bar=False,
                      on_step=True,
                      batch_size=classes.shape[0])
        self.log(f'{log_prefix}/loss', loss.sum(), prog_bar=True, on_step=True, batch_size=classes.shape[0])

        return loss.sum()


    def training_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'validation')
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.shared_step(batch, 'test')
        return loss


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

        total_loss = v_loss + a_loss
        partial_loss_dict = dict(v_loss=v_loss,
                                 a_loss=a_loss,
                                 v_loss_positive=v_loss_positive,
                                 v_loss_negative=v_loss_negative,
                                 a_loss_positive=a_loss_positive,
                                 a_loss_negative=a_loss_negative)

        return total_loss, partial_loss_dict



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

        total_loss = (loss_positive + loss_negative)
        partial_loss_dict = dict(loss_positive=loss_positive,
                                 loss_negative=loss_negative)
        return total_loss, partial_loss_dict

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
        self.save_hyperparameters()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=cache_dir).to(device=self.device)

        # self.model.config.vision_config.video_decode_backend = 'pytorchvideo'

        # self.modality_transform = lb.LanguageBindVideoProcessor(self.model.config)

        self.model.text_model = None
        self.model.text_projection = None

    def forward(self, input):
        # with torch.no_grad():
            # input = self.process_input_video(input_paths, clip_start_times, clip_end_times)
        output = self.model.vision_model(pixel_values=input)[1]
        output_projected = self.model.visual_projection(output)
        return output_projected

    # def process_input_video(self, input_paths, clip_start_times, clip_end_times):
    #     """
    #     We do this instead of using VideoProcessor.__call__ so we can supply the start and end times
    #     """
    #     processor = self.modality_transform

    #     # TODO this is very slow
    #     pixel_values = [processor.image_processor(input_path,
    #                     processor.transform,
    #                     video_decode_backend='pytorchvideo',
    #                     clip_start_sec=clip_start_time,
    #                     clip_end_sec=clip_end_time,
    #                     num_frames=None)['video']
    #     for input_path, clip_start_time, clip_end_time
    #     in zip(input_paths, clip_start_times, clip_end_times)]

    #     return torch.stack(pixel_values).to(device=self.device)

    def parameters(self):
        return list(self.model.vision_model.parameters()) + list(self.model.visual_projection.parameters())


class LB_AudioEncoder(pl.LightningModule):
    """
    TODO: This model also contains the CLIP text encoder, which is loaded multiple times
    May take up a lot more GPU space
    """

    def __init__(self,
                 pretrained_ckpt: str = 'LanguageBind/LanguageBind_Audio_FT',
                 cache_dir: str = './cache_dir'):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
        # self.modality_transform = lb.LanguageBindAudioProcessor(self.model.config)

        self.model.text_model = None
        self.model.text_projection = None

    def forward(self, input):
        # with torch.no_grad():
            # input = self.process_input_audio(input_paths, start_times, end_times)
        output = self.model.vision_model(pixel_values=input)[1]
        output_projected = self.model.visual_projection(output)
        return output_projected


    # def process_input_audio(self, input_paths, start_times, end_times):
    #     processor = self.modality_transform

    #     pixel_values = []
    #     for input_path, start_time, end_time in zip(input_paths, start_times, end_times):
    #         waveform, sample_rate = lb.audio.processing_audio.torchaudio_loader(input_path)
    #         start_frame = int(start_time * sample_rate)
    #         end_frame = int(end_time * sample_rate)
    #         waveform = waveform[:, start_frame : end_frame]

    #         pixel_values.append(processor.transform((waveform, sample_rate)))

    #     return torch.stack(pixel_values).to(device=self.device)

    def parameters(self):
        return list(self.model.vision_model.parameters()) + list(self.model.visual_projection.parameters())


class LB_LabelEncoder(pl.LightningModule):
    """
    TODO: This model also contains the CLIP video encoder, which is loaded multiple times
    May take up a lot more GPU space
    """

    def __init__(self,
                 pretrained_ckpt: str = 'LanguageBind/LanguageBind_Video_FT',
                 cache_dir: str = './cache_dir'):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_ckpt = pretrained_ckpt
        self.model = lb.LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
        self.tokenizer = lb.LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)

        self.model.vision_model = None
        self.model.visual_projection = None

    @torch.no_grad()
    def forward(self, text):
        tokens_mask = self.tokenizer(text,
                                     max_length=77,
                                     padding='max_length',
                                     return_tensors='pt',
                                     truncation=True).to(device=self.device)

        output = self.model.text_model(**tokens_mask)[1]
        output_projected = self.model.text_projection(output)

        return output_projected

    
class LB_VideoEncoder_PartiallyFrozen(LB_VideoEncoder):
    def __init__(self,
                 n_finetune_layers: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Freeze the entire vision model
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

        # Unfreeze the last N encoder layers
        for layer in self.model.vision_model.encoder.layers[-n_finetune_layers:]:
            param.requires_grad = True

        # Unfreeze the final layernorm
        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True


class LB_AudioEncoder_PartiallyFrozen(LB_AudioEncoder):
    def __init__(self,
                 n_finetune_layers: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Freeze the entire vision model
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

        # Unfreeze the last N encoder layers
        for layer in self.model.vision_model.encoder.layers[-n_finetune_layers:]:
            param.requires_grad = True

        # Unfreeze the final layernorm
        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True

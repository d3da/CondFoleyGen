import torch
import torch.nn.functional as F

class LabelInfoNCELoss(torch.nn.Module):
    def __init__(self,
                 softmax_temperature):
        super().__init__()
        self.softmax_temperature = softmax_temperature

    def forward(self,
                encoder_embeddings,  # B x D
                class_nums,  # B
                label_embeddings):  # L x D

        sim_matrix = F.cosine_similarity(encoder_embeddings.unsqueeze(1),
                                         label_embeddings.unsqueeze(0),
                                         dim=-1)
        sim_matrix /= self.softmax_temperature
        # B x L
        prob_logits = F.log_softmax(sim_matrix, dim=-1)

        classes_one_hot = F.one_hot(class_nums,
                                    num_classes=label_embeddings.shape[0])
        # B x L

        loss = -prob_logits.mul(classes_one_hot).sum(dim=1).mean()
        return loss


class TemporalInfoNCELoss(torch.nn.Module):
    def __init__(self,
                 softmax_temperature):
        super().__init__()
        self.softmax_temperature = softmax_temperature

    def forward(self,
                a_encoder_embeddings,  # B x Ca x D
                v_encoder_embeddings):  # B x Cv x D
        batch_size, segments_per_clip, _ = a_encoder_embeddings.shape

        sim_matrix = F.cosine_similarity(a_encoder_embeddings.unsqueeze(2),  # B x Ca x 1 x D
                                         v_encoder_embeddings.unsqueeze(1),  # B x 1 x Cv x D
                                         dim=-1)  # B x Ca x Cv
        sim_matrix /= self.softmax_temperature
        identity = torch.eye(segments_per_clip, device=sim_matrix.device)

        audio_prob_logits = F.log_softmax(sim_matrix, dim=2)
        video_prob_logits = F.log_softmax(sim_matrix, dim=1)

        audio_loss = -audio_prob_logits.mul(identity).sum(dim=-1).sum(dim=-1).mean()
        video_loss = -video_prob_logits.mul(identity).sum(dim=-1).sum(dim=-1).mean()
        return audio_loss + video_loss

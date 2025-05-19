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

        loss = -prob_logits.mul(classes_one_hot).sum(dim=1)  # B
        return loss.mean()


class TemporalInfoNCELoss(torch.nn.Module):
    def __init__(self,
                 softmax_temperature):
        super().__init__()
        self.softmax_temperature = softmax_temperature

    def forward(self,
                a_encoder_embeddings,
                v_encoder_embeddings):
        # TODO
        return torch.tensor(0)

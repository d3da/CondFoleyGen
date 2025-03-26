import torch
import torch.nn.functional as F

from foley2025.utils.similarity_function import SimilarityFunction

class AlignFunction(torch.nn.Module):
    def __init__(self, conf_similarity_str, softmax_temperature, *args):
        super().__init__()
        self.similarity_fn = SimilarityFunction.from_string(conf_similarity_str)
        self.softmax_temperature = softmax_temperature

    def forward(self,
                tcc_audio_embeddings,  # .. x a_length x dimension
                video_embeddings):  # .. x v_length x dimension
        assert tcc_audio_embeddings.shape[-1] == video_embeddings.shape[-1]

        sim_matrix = self.similarity_fn(tcc_audio_embeddings, video_embeddings)  # .. x L_a x L_v
        sim_weights = F.softmax(sim_matrix / self.softmax_temperature, dim=-1)
        sim_weights_normalized = sim_weights / sim_weights.sum(dim=-2, keepdim=True)
        # Alternative would be to take the softmax over the audio dimension. No normalization needed then.

        sim_broadcast = sim_weights_normalized.unsqueeze(-1)  # .. x L_a x L_v x 1
        audio_emb_broadcast = tcc_audio_embeddings.unsqueeze(-2)  # .. x L_a x 1 x dim
        aligned_audio = audio_emb_broadcast.mul(sim_broadcast).sum(dim=-2)  # .. x L_v x dim
        return aligned_audio


if __name__ == '__main__':
    d = 'cuda'
    f = AlignFunction()

    x = torch.rand((10, 768), device=d)
    y = torch.rand((12, 768), device=d)

    print(f(x,y))
    import pdb; pdb.set_trace()

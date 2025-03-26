import torch


class SimilarityFunction(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
        raise NotImplementedError

    @staticmethod
    def from_string(conf_string):
        if conf_string == 'cosine':
            return CosineSimilarityFunction()
        elif conf_string == 'negative-euclidean':
            return NegativeEuclideanSimilarityFunction()
        raise NotImplementedError(f'Unsupported similarity function: {conf_string}')


class CosineSimilarityFunction(SimilarityFunction):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
        assert len(u.shape) == len(v.shape)
        assert u.shape[-1] == v.shape[-1]

        if len(u.shape) == 2:
            return self._similarity_unbatched(u, v)
        elif len(u.shape) == 3:
            return self._similarity_batched(u, v)

        raise ValueError(f'Unsupported shape: {u.shape}, {v.shape}')

    def _similarity_unbatched(self, u, v):
        # U: L_u x D
        # V: L_v x D
        return torch.nn.functional.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=-1)
        # Output: L_u x L_v

    def _similarity_batched(self, u, v):
        assert u.shape[0] == v.shape[0]
        # U: B x L_u x D
        # V: B x L_v x D
        return torch.nn.functional.cosine_similarity(u.unsqueeze(2), v.unsqueeze(1), dim=-1)
        # Output: B x L_u x L_v


class NegativeEuclideanSimilarityFunction(SimilarityFunction):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
        assert len(u.shape) == len(v.shape)
        assert u.shape[-1] == v.shape[-1]
        if len(u.shape) == 3:
            assert u.shape[0] == v.shape[0]

        return -1 * torch.cdist(u, v, p=2)



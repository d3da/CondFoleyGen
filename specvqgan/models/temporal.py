import random

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class TCCLoss(pl.LightningModule):

    def __init__(self, tcc_lambda):
        super().__init__()

        self.tcc_lambda = tcc_lambda

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.softmax = torch.nn.Softmax(dim=0)

    def tcc_similarities(self,
                         u,  # L_1 x D
                         v):  # L_2 x D
        out_sim = F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=2)  # L_i x L_2
        alphas = F.softmax(out_sim, dim=1)  # L_i x L_2
        snn = alphas.matmul(v)  # L_i x D
        in_sim = F.cosine_similarity(snn.unsqueeze(1), u.unsqueeze(0), dim=2)  # L_2
        betas = F.softmax(in_sim, dim=1)  # L_1 x L_1
        return betas

    def regression_loss(self,
                        u,  # L_1 x D
                        v):  # L_2 x D
        betas = self.tcc_similarities(u, v)  # L_1 x L_1
        arange = torch.arange(betas.shape[0]).to(betas.device)  # L_1
        mean_idx = betas.mul(arange.unsqueeze(0)).sum(dim=1)  # L_1
        variance = betas.matmul(torch.square(arange - mean_idx)) # L_1
        loss = torch.square(arange - mean_idx) / variance \
            + 0.5 * self.tcc_lambda * torch.log(variance)
        return loss  # L_1


class GTCCLoss(pl.LightningModule):
    def __init__(self,
                 n_components: int,
                 lbfgs_max_iters: int,
                 lbfgs_lr: float,
                 tcc_lambda: float,
                 window_ratio: float,
                 divide_by_variance: bool):
        super().__init__()

        self.n_components = n_components
        self.lbfgs_max_iters = lbfgs_max_iters
        self.lbfgs_lr = lbfgs_lr
        self.tcc_lambda = tcc_lambda
        self.window_ratio = window_ratio
        self.divide_by_variance = divide_by_variance

    def component_snns(self,
                       secondary_sequence,  # L_2 x D
                       component_means,  # L_1 x K
                       component_variances):  # L_1 x K
        sequence_length = secondary_sequence.shape[0]
        component_probabilities = self.component_probabilities(sequence_length,
                                                               component_means,
                                                               component_variances)   # L_1 x K x L_2
        snns = torch.einsum('ikj, jd -> ikd', component_probabilities, secondary_sequence)  # L_1 x K x D
        return snns

    def stochastic_window_mask(self, sequence_length):
        """
        Diff with GTCC:
        - Randomly select shift out of [0, ..., window_size - 1] instead of [0, window_size - 1]
        - Apply 'insurance' fix only where needed, so that the window size is not increased for outer indices
        """
        ones = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=self.device)
        if self.window_ratio >= 1.0:
            return ones

        window_size = round(self.window_ratio * sequence_length)
        shift = random.randint(0, window_size - 1)

        upper_triangle = torch.triu(ones, diagonal=-shift)
        lower_triangle = torch.triu(ones, diagonal=1 + shift - window_size).T
        windows = torch.logical_and(upper_triangle, lower_triangle)

        # Fix windows shifted so far that the window size decreased
        if shift > 0:
            windows[:shift, :window_size] = True
        if shift < window_size - 1:
            windows[1 + shift - window_size:, -window_size:] = True

        return windows  # L x L


    def gtcc_loss(self,
                  u,  # L_1 x D
                  v): # L_2 x D
        out_sim = F.cosine_similarity(u.unsqueeze(1), v.unsqueeze(0), dim=2)  # L_i x L_2
        # out_sim = -torch.sqrt(2 - 2 * out_sim + 1e-6)

        alphas = F.softmax(out_sim, dim=-1)  # L_i x L_2
        component_means, component_variances, component_weights = self.fit_gmm(alphas)  # L_i x K
        snns = self.component_snns(v, component_means, component_variances)  # L_i x K x D

        in_sim = F.cosine_similarity(snns.unsqueeze(1),  # L_i x 1 x K x D
                                     u.unsqueeze(1).unsqueeze(0),  # 1 x L_1 x 1 x D
                                     dim=-1)  # L_i x L_1 x K
        # in_sim = -torch.sqrt(2 - 2 * in_sim + 1e-6)

        mask = self.stochastic_window_mask(in_sim.shape[0])  # L_i x L_1
        in_sim = in_sim.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        betas = F.softmax(in_sim, dim=1)

        arange = torch.arange(betas.shape[0], device=self.device, dtype=self.dtype)  # L_1
        mean_idx = torch.einsum('iuk, u -> ik', betas, arange)  # L_i x K
        error = arange.unsqueeze(1).unsqueeze(0) - mean_idx.unsqueeze(1)  # (1 x L_1 x 1) - (L_i x 1 x K) -> L_i x L_1 x K
        variances = torch.einsum('iuk, iuk -> ik', betas, torch.square(error))  # L_i x K

        component_losses = torch.square(arange.unsqueeze(1) - mean_idx)  # L_i x K
        if self.divide_by_variance:  # TODO: GTCC code doesn't do this but the paper mentions it. It seems unnecessary
            component_losses /= variances
        component_losses += 0.5 * self.tcc_lambda * torch.log(variances)

        weighted_losses = torch.einsum('ik, ik -> i', component_losses, component_weights)
        return weighted_losses


    def fit_gmm(self,
                alphas):  # L_i x L_2
        orig_seq_length = alphas.shape[0]
        sequence_length = alphas.shape[1]

        step = sequence_length / self.n_components
        component_means = torch.arange(start=step / 2,
                                       end=sequence_length + step / 2,
                                       step=step,
                                       device=self.device) \
            .unsqueeze(0) \
            .expand(orig_seq_length, self.n_components) \
            .clone() \
            .requires_grad_(True)  # L_i x K

        component_log_variances = torch.zeros_like(component_means, requires_grad=True)  # L_i x K
        component_weight_logits = torch.zeros_like(component_means, requires_grad=True)  # L_i x K

        weights = (component_means, component_log_variances, component_weight_logits)
        optimizer = torch.optim.LBFGS(params=weights,
                                      max_iter=self.lbfgs_max_iters,
                                      lr=self.lbfgs_lr)

        def _closure():
            optimizer.zero_grad()
            with torch.enable_grad():
                parameters = self.transform_parameters(*weights)
                loss = self.kl_divergence(alphas, *parameters)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(_closure)
        return self.transform_parameters(*weights)

    def transform_parameters(self, component_means, component_log_variances, component_weight_logits):
        return (component_means,
                torch.exp(component_log_variances),
                F.softmax(component_weight_logits, dim=-1))

    def component_probabilities(self,
                                sequence_length,
                                component_means,  # L_i x K
                                component_variances):  # L_i x K
        component_means = component_means.clamp(min=-1, max=sequence_length)

        arange = torch.arange(0, sequence_length, device=self.device)  # L_2
        dst = torch.square(arange.unsqueeze(0).unsqueeze(0)
                           - component_means.unsqueeze(-1))  # L_i x K x L_2
        gaussian_logits = -0.5 * dst / component_variances.unsqueeze(2)
        gaussian_probabilities = F.softmax(gaussian_logits, dim=2)
        return gaussian_probabilities  # L_1 x K x L_2


    def gmm_probabilities(self,
                          sequence_length,
                          component_means,  # L_i x K
                          component_variances,  # L_i x K
                          component_weights):  # L_i x K
        """
        Calculate the probability distribution of a discrete Gaussian Mixture Model with K components
        """
        component_probabilities = self.component_probabilities(sequence_length,
                                                               component_means,
                                                               component_variances)  # L_1 x K x L_2
        probabilities = torch.einsum('ikj, ik -> ij', component_probabilities, component_weights)  # L_1 x L_2
        return probabilities

    def kl_divergence(self,
                      alphas,  # L_1 x L_2
                      component_means,  # L_1 x K
                      component_variances,  # L_1 x K
                      component_weights,  # L_1 x K
                      epsilon=1e-30):
        sequence_length = alphas.shape[1]
        gmm_probabilities = self.gmm_probabilities(sequence_length,
                                                   component_means,
                                                   component_variances,
                                                   component_weights)  # L_1 x L_2
        log_gmm = torch.log(gmm_probabilities + epsilon)  # L_1 x L_2
        log_alpha = torch.log(alphas + epsilon)  # L_1 x L_2
        log_likelihood_ratio = log_gmm - log_alpha  # L_1 x L_2
        kl_divergence = gmm_probabilities * log_likelihood_ratio - alphas * log_likelihood_ratio  # L_1 x L_2
        return kl_divergence.sum()  # scalar


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../GTCC_CVPR2024/')
    import utils.loss_functions as GTCC_losses

    L = GTCCLoss(n_components=4, lbfgs_lr=0.5, lbfgs_max_iters=500, tcc_lambda=0.05, divide_by_variance=False, window_ratio=0.25)

    u = torch.tensor([[0.2,0.6,0.2],
                     [0.5,0.2,0.6],
                      [0.8,0.9,0.2],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                     [0.7,0.2,1.5]])
    v = torch.tensor([[0.2,0.6,0.2],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,20.5,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.5,0.2,0.6],
                      [0.7,0.2,1.5]])
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    print('--------------------------------')
    print('Ours (batched): ', L.gtcc_loss(u, v).sum())
    print('--------------------------------')
    print('Paper:', GTCC_losses.GTCC_loss([u, v], n_components=4, gamma=1, delta=0.25, alignment_variance=0.05, max_gmm_iters=500))

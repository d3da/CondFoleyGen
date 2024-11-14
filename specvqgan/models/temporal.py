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
    def __init__(self, n_components, lbfgs_max_iters, lbfgs_lr, tcc_lambda):
        super().__init__()

        self.n_components = n_components
        self.lbfgs_max_iters = lbfgs_max_iters
        self.lbfgs_lr = lbfgs_lr
        self.tcc_lambda = tcc_lambda

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.softmax = torch.nn.Softmax(dim=0)

    def gtcc_loss(self,
                  i,
                  u,  # L_1 x D
                  v): # L_2 x D
        """
        # TODO: Batched version of the algorithm, or at least consider multiple i in calculating alphas.
        #       Fitting the GMM still done by looping through all i's
        """
        u_i = u[i]  # D
        out_sim = self.cos_sim(u_i.unsqueeze(0), v)  # L_2
        alphas = self.softmax(out_sim)  # L_2

        component_means, component_variances, component_weights = self.fit_gaussian_mixture_model(alphas)
        snns = self.component_snns(v, component_means, component_variances)  # K x D

        in_sim = torch.nn.functional.cosine_similarity(u.unsqueeze(1),  # L_1 x 1 x D
                                                       snns.unsqueeze(0),  # 1 x K x D
                                                       dim=2)  # (L_1 x K x D) -> L_1 x K
        betas = self.softmax(in_sim)  # L_1 x K

        # TODO: Set beta_j = 0 for j outside a window w around i
        #       Do we need to do that before normalizing betas?

        arange = torch.arange(0, betas.shape[0], device=self.device).unsqueeze(1)  # L x 1

        # TODO don't use mul().sum() but use torch.einsum('kl, k -> k', betas, arange(unsqueezed))
        mean_idxs = betas.mul(arange).sum(dim=0)  # K
        variances = torch.square(arange - mean_idxs.unsqueeze(0)).mul(betas).sum(dim=0)  # K  # TODO use einsum/dot

        component_losses = torch.square(i - mean_idxs) / variances + 0.5 * self.tcc_lambda * variances  # K
        return component_losses.mul(component_weights).sum()  # TODO use dot product


    def component_snns(self,
                       primary_sequence,  # L_1 x D
                       component_means,
                       component_variances):
        sequence_length = primary_sequence.shape[0]
        component_probabilities = self.component_probabilities(sequence_length, component_means, component_variances)  # L x K
        neighbor_weights = primary_sequence.unsqueeze(1).mul(component_probabilities.unsqueeze(2))  # L x K x D
        snns = neighbor_weights.sum(dim=0)  # K x D  # TODO use einsum
        return snns

    def component_probabilities(self,
                                sequence_length,
                                component_means,
                                component_variances):
        """
        Calculate K per-component probability distributions of a mixture of discrete gaussian distributions.
        """
        component_means = component_means.clamp(0, sequence_length)  # This seems necessary to stabilize the optimization
        component_variances = torch.abs(component_variances)  # K   # TODO move this?
        # TODO clamp the variances to a minimal of 0.5 like original authors

        arange = torch.arange(0, sequence_length, step=1, device=self.device)  # L
        dst = torch.square(arange.unsqueeze(1) - component_means.unsqueeze(0))  # (L x 1) - (1 x K) -> L x K
        gaussian_logits = -0.5 * dst / component_variances.unsqueeze(0)  # L x K
        gaussian_probabilities = self.softmax(gaussian_logits)  # L x K
        return gaussian_probabilities

    def gmm_probabilities(self,
                          sequence_length,
                          component_means,
                          component_variances,
                          component_weights):
        """
        Calculate the probability distribution of a discrete Gaussian Mixture Model with K components
        """
        component_probabilities = self.component_probabilities(sequence_length, component_means, component_variances)
        weighted_probabilities = component_weights.unsqueeze(0) * component_probabilities  # L x K
        return weighted_probabilities.sum(dim=1)  # L  # TODO use einsum?


    @torch.enable_grad()
    def kl_divergence(self, alphas, component_means, component_variances, component_weights, epsilon=1e-30):
        """
        Calculate the KL-divergence between a given discrete probability distribution `alphas`
        and a discrete Gaussian Mixture Model.

        A symmetric version of KL-divergence is used: D = D_{KL}(P || Q) + D_{KL}(Q || P)
        """
        gmm_probabilities = self.gmm_probabilities(alphas.shape[0], component_means, component_variances, component_weights)
        log_gmm = torch.log(gmm_probabilities + epsilon)  # L
        log_alpha = torch.log(alphas + epsilon)  # L
        log_likelihood_ratio = log_gmm - log_alpha
        kl_divergence = gmm_probabilities * log_likelihood_ratio - alphas * log_likelihood_ratio  # L
        # print('--------------------')
        # print(kl_divergence)
        # print(alphas)
        # print(gmm_probabilities)
        # print(gmm_probabilities.sum())
        # print(component_means)
        # print(component_weight_logits)
        # print(variances)
        # import pdb; pdb.set_trace()
        return kl_divergence.sum()

    def fit_gaussian_mixture_model(self, alphas):
        sequence_length = alphas.shape[0]
        component_means = torch.arange(0,
                                       sequence_length,
                                       step=sequence_length / self.n_components,
                                       device=self.device,
                                       requires_grad=True)  # K
        component_variances = torch.ones_like(component_means, requires_grad=True)  # K, all values 1
        component_weight_logits = torch.zeros_like(component_means, requires_grad=True)  # K, all values initialized as 0

        optimizer = torch.optim.LBFGS(params=[component_means, component_variances, component_weight_logits],
                                      max_iter=self.lbfgs_max_iters,
                                      lr=self.lbfgs_lr)

        component_weights = self.softmax(component_weight_logits)  # Enforce a probability distribution

        def _closure():
            optimizer.zero_grad()
            loss = self.kl_divergence(alphas, component_means, component_variances, component_weights)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(_closure)

        return component_means, component_variances, component_weights


if __name__ == '__main__':
    # L = GTCCLoss(n_components=50, lbfgs_max_iters=500, lbfgs_lr=0.01, tcc_lambda=0.5)
    L = TCCLoss(tcc_lambda=0.5)

    # alphas = torch.tensor([0.2, 0.3, 0.1, 0.6, 0.9, 0.4, 0.2, 0., 0.0])
    # alphas = torch.tensor([0.])
    # x = L.fit_gaussian_mixture_model(torch.nn.functional.softmax(alphas))
    # print(x)

    u = torch.rand([9, 256])
    v = torch.rand([1, 256])
    i = 3
    # q = L.gtcc_loss(i, u, v)
    q = L.loss_regression(i, u, v)

    print(q)

    # import pdb; pdb.set_trace()

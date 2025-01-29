import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pytorch_lightning as pl


class TCCLoss(pl.LightningModule):
    def __init__(self, tcc_lambda, softmax_temperature):
        super().__init__()

        self.alignment_variance = tcc_lambda
        self.softmax_temp = softmax_temperature
        self.normalizer = nn.Softmax(dim=0)

    def forward(self, sequences):
        # loss term 2 is TCC loss
        tcc_count = 0
        all_tcc_losses = None
        ###################################################
        ### iterate through primary sequences
        for i, primary in enumerate(sequences):
            primary = primary.to(self.device)
            idx_range = torch.arange(0, primary.shape[0], dtype=torch.float).to(self.device).detach()
            ###################################################
            ### iterate through secondary sequences
            for j, secondary in enumerate(sequences):
                secondary = secondary.to(self.device)
                #################################
                # get intermittent variables
                ALPHA = self.normalizer(-torch.cdist(primary, secondary, p=2).T / self.softmax_temp).T
                SNN = ALPHA @ secondary
                BETA = self.normalizer(-torch.cdist(SNN, primary, p=2).T / self.softmax_temp).T
                mus = BETA @ idx_range
                variances = BETA @ torch.square(idx_range - mus)
                
                #################################
                # get TCC loss
                loss_terms = torch.divide(torch.square(idx_range - mus), variances) + self.alignment_variance * torch.log(torch.sqrt(variances) + 1e-20)
                
                if all_tcc_losses is None:
                    all_tcc_losses = torch.sum(loss_terms) 
                else: 
                    all_tcc_losses += torch.sum(loss_terms)
                tcc_count += 1
        return all_tcc_losses / tcc_count


class LegacyGTCC(pl.LightningModule):
    def __init__(self, n_components, gamma, delta, dropouts, softmax_temp, alignment_variance, max_gmm_iters, epoch):
        super().__init__()

        self.n_components = n_components
        self.gamma = gamma
        self.delta = delta
        self.dropouts = dropouts

        self.softmax_temp = softmax_temp
        self.alignment_variance = alignment_variance
        self.max_gmm_iters = max_gmm_iters
        self.epoch = epoch


def GTCC_loss(self, sequences):
    assert .05 <= self.delta <= 1
    tiny_number = 0
    all_tcc_losses = None
    drop_min = self.gamma**(self.epoch + 1)
    ###################################################
    ### iterate through primary sequences
    for i, primary in enumerate(sequences):
        primary = primary.to(self.device)
        N = primary.shape[0]
        margin = round(N * self.delta)

        max_comparisons = 20
        indices_to_check = torch.randperm(N)[:max_comparisons].to(self.device).sort().values
        ind_bool = torch.zeros(N, dtype=torch.bool).to(self.device)
        ind_bool[indices_to_check] = True
        del indices_to_check
        idx_range = torch.arange(0, N, dtype=torch.float).to(self.device).detach()
        indbool_idx_range = idx_range[ind_bool]
        margin_identity = create_stochastic_margin_identity_matrix(N, margin).to(self.device)[ind_bool]

        ###################################################
        ### iterate through secondary sequences
        for j, secondary in enumerate(sequences):
            if i == j: # skip if same sequence
                continue
            M = secondary.shape[0]
            # get the drop vectors!
            if self.gamma < 1:
                BX = primary @ self.dropouts[j][:-1].squeeze() + self.dropouts[j][-1]
                BX = (BX - BX.mean()) / BX.std()
                BX = drop_min + (1-drop_min) * nn.Sigmoid()(BX)[ind_bool]
            
            secondary = secondary.to(self.device)
            cdist = torch.cdist(primary, secondary, p=2)[ind_bool]
            ALPHA_exp = torch.exp(-cdist / self.softmax_temp) + tiny_number
            ALPHA = (ALPHA_exp.T / (ALPHA_exp.sum(dim=1))).T
            
            gmm = torch.zeros((ALPHA.shape[0], self.n_components, M)).to(self.device)
            spread = torch.zeros((ALPHA.shape[0], self.n_components)).to(self.device)
            for u in range(ALPHA.shape[0]):
                start = time.time()
                g, s, _ = get_gmm_lfbgf(
                    ALPHA[u], self.n_components, max_iters=self.max_gmm_iters
                )
                g = g.to(self.device)
                s = s.to(self.device)

                gmm[u] = g
                spread[u] = s
                
            SNNs = (gmm @ secondary)
            prim_expanded = primary.unsqueeze(0)
            snn_cdist = torch.cdist(SNNs, prim_expanded, p=2)

            BETAs = (margin_identity * torch.exp(-snn_cdist / self.softmax_temp).permute(1,0,2)).permute(1, 0, 2) + tiny_number
            BETAs = (BETAs.permute(2, 0, 1) / (BETAs.sum(dim=2) + 1e-6)).permute(1, 2, 0)

            mus = (BETAs @ idx_range).unsqueeze(-1)
            
            variances = torch.sum(BETAs * torch.square(idx_range - mus), dim=2)

            index_margin_identity = idx_range * margin_identity
            
            spread = spread.to(self.device)
            for t in range(max_comparisons):
                idx_mask = index_margin_identity[t]
                set_of_mus = mus[t]
                set_of_vars = variances[t]
                this_spread = spread[t]
                idx_mask = idx_mask[margin_identity[t].bool()].unsqueeze(0)

                each_mu_error = torch.square(indbool_idx_range[t] - set_of_mus).squeeze()
                if self.alignment_variance > 0:
                    tcc = each_mu_error + self.alignment_variance * torch.log(torch.sqrt(set_of_vars) + 1e-20)
                else:
                    tcc = each_mu_error

                if contains_non_float_values(tcc):
                    print("contains_non_float_values(tcc)")
                    print(set_of_vars)
                    print(set_of_mus)
                    exit(1)
                align_loss = torch.inner(tcc, this_spread)
                if contains_non_float_values(1/align_loss):
                    print("contains_non_float_values(1/align_loss)")
                    print(align_loss)
                    exit(1)

                if self.gamma < 1:
                    tcc_loss_term = BX[t] * align_loss + (1-BX[t]) * (1 / align_loss)
                else:
                    tcc_loss_term = align_loss

                if None in [all_tcc_losses]:
                    all_tcc_losses = tcc_loss_term
                else: 
                    all_tcc_losses += tcc_loss_term

    return all_tcc_losses


#########################################
# below function is only for GTCC
#########################################
def create_stochastic_margin_identity_matrix(size, width):
    m = int(np.random.choice([0, 1]) * width)
    mm = 1 - torch.triu(torch.ones(size, size), diagonal=m+1).T
    wm = torch.triu(torch.ones(size, size), diagonal=width - m)
    insurance = torch.zeros((size, size))
    insurance[:width, :width] = 1
    insurance[-width:, -width:] = 1
    insurance = insurance.bool()
    return torch.logical_or(torch.logical_xor(wm, mm), insurance).float()




def get_gmm_lfbgf(
        probability,
        n_components=5,
        device='cpu',
        debug=False,
        max_iters=50,
        history_size=10,
        max_iter=4,
        loss_fn='KL'
    ):
    #################################
    ### Loss functions
    #################################
    def kl_divergence_loss(mus, sigmas, psis):
        all_gaussians = get_gaussians(mus=mus, vars=sigmas)
        gmm = get_spread(spread_vals=psis) @ all_gaussians
        gmm_log = torch.log(gmm + 1e-30)
        p_log = torch.log(probability + 1e-30)

        kl_div = torch.sum(torch.sum(probability * (p_log - gmm_log))) + \
            torch.sum(torch.sum(gmm * (gmm_log - p_log)))
        p_log_diff = torch.diff(p_log)
        intensity_div = torch.abs(
            (p_log_diff / p_log_diff.max() + p_log[:-1] / p_log.max()) * ((p_log - gmm_log)[:-1] + (p_log - gmm_log)[1:])
        ).sum()

        if kl_div < 0 or contains_non_float_values(kl_div):
            print(probability.sum())
            raise Exception
        return kl_div

    def get_spread(spread_vals):
        return softmax(spread_vals)
    
    def get_gaussians(mus, vars):
        stds = torch.sqrt(torch.clamp(vars.view(-1, 1), .5))
        mus = torch.clamp(mus.view(-1, 1), 0, N-1)
        all_gaussians = torch.exp(
            -0.5 * (torch.subtract(arange, mus)  / stds) ** 2
        ) / (
            stds * (2 * torch.pi) ** 0.5
        )
        all_gaussians = torch.divide(all_gaussians.T, all_gaussians.sum(dim=1)).T
        return all_gaussians
    
    def closure():
        lbfgs.zero_grad()
        if loss_fn == 'KL':
            objective = kl_divergence_loss(means, stds, spreads)
        else:
            print('ERROR')
            exit(1)
        if type(objective) == tuple:
            return objective
        objective.backward()
        return objective
    try:
        with torch.enable_grad():
            probability = probability.to(device).detach()
            dtype = torch.float32
            N = probability.shape[0]
            arange = torch.arange(N, dtype=dtype).to(device).detach()
            start = time.time()
            softmax = nn.Softmax(dim=0).to(device)

            means = torch.nn.Parameter(torch.tensor(
                    [1 + i * (N-2)/n_components for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            stds = torch.nn.Parameter(torch.tensor(
                    [N for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            spreads = torch.nn.Parameter(torch.tensor(
                    [1 for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            lbfgs = optim.LBFGS(
                [
                    {'params': [means, stds, spreads]}
                ],
                history_size=history_size,
                max_iter=max_iter, 
                line_search_fn="strong_wolfe",
                lr=.5
            )
            
            s = time.time()
            for _ in range(max_iters):
                lbfgs.step(closure)
            if debug:
                speed = time.time() - s
                return get_gaussians(means, stds), get_spread(spreads), {
                    'KL': (kl_divergence_loss(means, stds, spreads).detach().item(), speed),
                    'mus': means,
                    'stds': stds,
                }
            else:
                return get_gaussians(means, stds), get_spread(spreads), means
    except Exception as e:
        traceback.print_exc()
        exit(1)
        return None, None, None


def contains_non_float_values(tensor):
    def check_tensor(data):
        # Check for NaN values
        nan_check = torch.isnan(data)
        
        # Check for positive infinity (inf) values
        pos_inf_check = torch.isinf(data)
        
        # Check for negative infinity (-inf) values
        neg_inf_check = torch.isinf(data) & (data < 0)
        
        # Combine the checks for NaN, inf, and -inf values
        has_non_float_values = torch.any(nan_check | pos_inf_check | neg_inf_check)
        
        return has_non_float_values.item()
    if torch.is_tensor(tensor):
        return check_tensor(tensor)
    elif isinstance(tensor, np.ndarray):
        return check_tensor(torch.from_numpy(tensor))
    elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], np.ndarray): # list of numpies
        return any([check_tensor(torch.from_numpy(one_array)) for one_array in tensor])
    elif isinstance(tensor, list) and len(tensor) > 0 and torch.is_tensor(tensor): # list of numpies
        return any([check_tensor(one_array) for one_array in tensor])
    elif isinstance(tensor, list) and len(tensor) > 0 and type(tensor[0]) == int: # list of numpies
        tensor = np.array(tensor)
        return check_tensor(torch.from_numpy(tensor))
    else:
        print("Bad input, must be (tensor, np.ndarray) or a list of either")
        exit(1)

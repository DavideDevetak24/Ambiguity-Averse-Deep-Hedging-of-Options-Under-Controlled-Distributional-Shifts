"""
In this module we define the inner risk measures to be used in the loss functions.

We propose three risk measures:
1. AADHInnerLoss: The inner loss function used in the AADH model
2. EntropicLoss: The entropic risk measure as defined in the Deep Hedging paper
3. CVaRLoss: The CVaR risk measure as defined in the Deep Hedging paper

We implement risk measures proposed in Buehler et al.'s Deep Hedging paper.
Entropic risk measure and CVaR. The one we use in the AADH model is the inner loss function 
defined as:

rho(X) = V(P&L_before_costs) + lambda * E(transaction_costs)

which is a mapping from R^(b*d) to R, where b is the batch size and d the number of assets
In our case d=1.

For (d=1), the loss function maps the P&L and transaction costs from clusters of paths ((R^l)) 
to a single scalar risk measure ((R)).
This is because we separate the paths in l clusters and compute the risk measure as inner loss.

"""

import torch
import torch.nn as nn


class AADHInnerLoss(nn.Module):
    """
    ρ_AADH(pl, tc) = V(pl) + λ * E(tc)

    Inputs: pl, tc tensors of shape [cluster_size, d] (d=1 in our case)
    Outputs: scalar tensor

    """

    def __init__(self, lam: float = 1/100):
        super().__init__()
        self.lam = float(lam)
    # pl_no_tc: total costs and tc: total costs
    def forward(self, pl_no_tc: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
        # pl, tc: [cluster_size, d]
        # correction for variance if there is only one path in the cluster
        if pl_no_tc.numel() > 1:
            V_pl_no_tc = torch.var(pl_no_tc, unbiased=True)
        else:
            V_pl_no_tc = torch.tensor(1e-5, device=pl_no_tc.device) # added variance floor
        E_tc = torch.mean(tc)
        return V_pl_no_tc + self.lam * E_tc


class EntropicLoss(nn.Module):
    """
    ρ_ent(pl) = (1/λ) * log E[ exp(-λ * pl) ]
    We minimize ρ_ent(pl). Lower is better.
    Input PL, batch size, output scalar

    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = float(lam)

    # tensor type can be removed, like "def forward(self, pl):"
    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        # pl: [batch]
        x = -self.lam * pl
        m = torch.max(x)    # makes computation safe (ensures training don't blow up)
        return (torch.log(torch.mean(torch.exp(x - m))) + m) / self.lam


class CVaRLoss(nn.Module):
    """
    OCE / CVaR-style objective with an auxiliary scalar w:

    L is the loss
    ρ_CVaR(L) = w + (1/(1-α)) * E[max(L - w, 0)]

    In other words: L = - pl

    If learn_w=True, w is optimized jointly with the model.

    """
    def __init__(self, alpha: float = 0.50, init_w: float = 0.0, learn_w: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        if learn_w == True:
            self.w = nn.Parameter(torch.tensor(float(init_w)))
        else:
            self.register_buffer("w", torch.tensor(float(init_w)))       

    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return (self.w + (1.0 / (1.0 - self.alpha))*torch.mean(torch.clamp( - pl - self.w, min=0.0)))  



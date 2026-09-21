"""
In this script we propose the outer risk measure.
As proposed in the paper "Ambiguity Averse Deep Hedging with Feature Clustering" by Jones et al. (2025),
we use an entropic risk measure.

rho(-R) = 1/alpha * log E_mu[exp(alpha * R)]

In other words, we input the inner loss function values calculated for each cluster,
and then we weight them according to the cluster probabilities mu.

To ensure numerical stability, we implement the log-sum-exp trick.

rho(-R) = r_max + 1/alpha * log E_mu[ exp(alpha * (R - r_max)) ]
where r_max = max R over clusters.

"""

import torch
import torch.nn as nn

class AADHOuterLoss(nn.Module):
    def __init__(self, alpha: float = 250):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, inner_losses: torch.Tensor, cluster_probs: torch.Tensor) -> torch.Tensor:
        """
        Inner losses shape : [num_clusters]
        Cluster probabilities shape : [num_clusters]
        """
        # max for numerical stability
        r_max = inner_losses.max()
        exponential = torch.exp(self.alpha * (inner_losses - r_max))
        # weighted expectation
        expectation = torch.sum(cluster_probs * exponential)
        outer_loss = r_max + (1.0 / self.alpha) * torch.log(expectation)
        
        return outer_loss

        

        

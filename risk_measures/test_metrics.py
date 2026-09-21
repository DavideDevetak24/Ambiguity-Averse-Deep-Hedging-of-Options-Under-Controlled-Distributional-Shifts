"""
In this script, we define the risk metrics used to evaluate the performance of hedging strategies.
We use an overall risk, given by the Outer Loss function, the variance on the P&L distribution,
and the total transaction costs incurred.

We also define the functions to compute the P&L array given the hedging strategy, the 
P&L summary statistics and a function to evaluate the performance of real data.

"""

# Adjust path for imports
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# import libraries
import numpy as np
import pandas as pd
import torch
# Import modules from other scripts
from utils.options import *
from utils.PL import comp_PL, comp_PL_components
from risk_measures.measures import AADHInnerLoss
# import modules for real data evaluation
from ambiguity_measures.KMeans_algo import compute_feature_matrix
from dataset.dataset import AADeepHedgingDataset, loader


# Validation function SDH
def test_result_metrics(model, loader, option_type_fn=eur_call_payoff, strike=100.0,
                        cost_rate=0.0, lam=1/100, device='cpu'):
    """
    Function to compute risk metrics on test data for a given hedging model.
    """
    model.eval()
    device = torch.device(device)
    running_overall_risk = torch.zeros((), device=device)
    running_overall_variance = torch.zeros((), device=device)
    running_transaction_costs = torch.zeros((), device=device)
    n = 0
    overall_risk_fn = AADHInnerLoss(lam=lam)
    with torch.inference_mode():
        for paths, _ in loader:
            paths = paths.to(device, non_blocking=True)
            batch_size = paths.size(0)

            delta = model(paths)

            PL_cluster, TC_cluster, PL_cluster_with_costs = comp_PL_components(
                delta,
                paths,
                payoff_fn=option_type_fn,
                payoff_kwargs={'strike': strike},
                cost_rate=cost_rate,
                p0=0.0
            )
            overall_risk = overall_risk_fn(PL_cluster, TC_cluster)
            
            # we compute the variance of the P&L batch with transaction costs
            overall_variance = torch.var(PL_cluster_with_costs)

            # we compute the total transaction costs
            transaction_costs = torch.sum(TC_cluster)

            running_overall_risk += overall_risk * batch_size
            running_overall_variance += overall_variance * batch_size
            # do not multiply by batch size since already summed above
            running_transaction_costs += transaction_costs

            n += batch_size
    overall_risk_metric = (running_overall_risk / n).item()
    overall_variance_metric = (running_overall_variance / n).item()
    overall_transaction_costs = (running_transaction_costs / n).item()

    return overall_risk_metric, overall_variance_metric, overall_transaction_costs



def compute_PL_array(model, loader, option_type_fn=eur_call_payoff, strike=100.0,
                      cost_rate=0.0, p0=0.0, device='cpu'):
    """
    Function to compute the P&L array for a given hedging model on the provided data loader.
    """
    model.eval()
    PL_list = []
    device = torch.device(device)
    with torch.inference_mode():
        for paths, _ in loader:
            paths = paths.to(device, non_blocking=True)

            delta = model(paths)

            PL_cluster = comp_PL(delta,
                                 paths,
                                 payoff_fn=option_type_fn,
                                 payoff_kwargs={'strike': strike},
                                 cost_rate=cost_rate,
                                 p0=p0,
                                 transaction_costs=True)
            PL_list.append(PL_cluster.cpu())

    PL_array = torch.cat(PL_list).numpy()
    return PL_array


# compute P&L summary statistics
def PL_summary(pl):
    """Return summary statistics as a dict"""
    return {
        "mean": float(np.mean(pl)),
        "std": float(np.std(pl)),
        "min": float(np.min(pl)),
        "median": float(np.quantile(pl, 0.5)),
        "max": float(np.max(pl)),
    }


# function to evalutate real data performance
def evaluate_real_data_performance(data_file_path, kmeans_model, 
                                   evaluation_model, strike, cost_rate, 
                                   lam, batch_size=100, device='cpu'):

    real_data = pd.read_csv(data_file_path, header=None).T
    
    real_data_feature_matrix = compute_feature_matrix(real_data)
    labels_real_data = kmeans_model.predict(real_data_feature_matrix)
    dataset_real_data = AADeepHedgingDataset(real_data, labels_real_data)

    real_data_loader, _, _ = loader(dataset_real_data, 
                                    batch_size,
                                    set_proportions = [1.0, 0.0, 0.0],
                                    shuffle=False)
    
    a, b, c = test_result_metrics(evaluation_model, 
                                  real_data_loader, 
                                  option_type_fn=eur_call_payoff, 
                                  strike=strike, 
                                  cost_rate=cost_rate, 
                                  lam=lam,
                                  device=device)
    
    return a, b, c









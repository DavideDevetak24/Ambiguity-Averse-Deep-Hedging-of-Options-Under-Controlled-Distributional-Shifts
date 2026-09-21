"""
In this script, we define utility functions for training, including
adjusting learning rates and calculating cluster probabilities from a DataLoader.

The functions are imported in train_loop_nn.py.
"""
import contextlib
import io

import numpy as np
import pandas as pd

# Adjustable learning rate function
def adjust_learning_rate(optimizer, epoch, starting_lr=0.001, schedule=None):
    """
    We create a customizable learning rate schedule.
    The schedule parameter is a list of tuples (epoch_threshold, divisor).
    
    We iterate through the schedule and check if the current epoch
    exceeds the epoch_threshold. If it does, we update the learning rate.
    
    """
    # Default schedule
    if schedule is None:
        schedule = [(0, 1), (50, 10), (100, 100), (250, 1000)]  # (epoch_threshold, divisor)
    
    # Find the appropriate divisor for the current epoch
    lr = starting_lr
    for e, divisor in schedule:
        if epoch >= e:  # Apply the divisor if the current epoch exceeds the threshold
            # In other words, we keep updating lr until we find the last applicable divisor
            lr = starting_lr / divisor
    # update the learning rates in the parameter groups
    # use a for loop to update all parameter groups (otherwise only the first group is updated)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def calculate_cluster_probabilities_loader(loader):
    """
    The function takes as input a dataloader and calculates the cluster probabilities
    The function detects automatically the unique clusters in the dataset and 
    returns a dictionary with the cluster probabilities in the format:
    {
        cluster_label: {
            "probability": prob
        },
        ...
    
    """
    # unwrap all labels from the dataloader
    label_batches = []
    for _, labels in loader:
        label_batches.append(labels.detach().cpu().numpy())
    labels = np.concatenate(label_batches)
    n_paths = len(labels)
    cluster_counts = np.bincount(labels)

    cluster_info = {}

    for cluster_label, count in enumerate(cluster_counts):
        probability = count / n_paths
        indices = np.where(labels == cluster_label)[0]

        cluster_info[cluster_label] = {
            "probability": probability,
            "path_indices": indices.tolist()
        }

    return cluster_info


def _mixture_grid(xi_step=0.1, xi_values=None):
    if xi_values is not None:
        values = [round(float(value), 10) for value in xi_values]
    else:
        if xi_step <= 0.0 or xi_step > 1.0:
            raise ValueError("xi_step must be in (0, 1].")

        values = []
        current = 0.0
        while current < 1.0:
            values.append(round(current, 10))
            current += xi_step
        values.append(1.0)

    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Mixture values must stay between 0 and 1.")

    return values


def _count_mixture_paths(n_paths, xi_heston):
    n_heston = int(round(float(xi_heston) * n_paths))
    n_heston = min(max(n_heston, 0), n_paths)
    n_garch = n_paths - n_heston
    return n_heston, n_garch


def _params_with_path_count(params, n_simulations, seed):
    params_out = dict(params)
    params_out["n_simulations"] = int(n_simulations)
    params_out["seed"] = int(seed)
    return params_out


def _generate_mixed_paths(heston_params, garch_params, n_paths, xi_heston, seed):
    from data_generators.heston import heston_data_generator
    from data_generators.garch import garch_data_generator

    n_heston, n_garch = _count_mixture_paths(n_paths, xi_heston)
    parts = []

    if n_heston > 0:
        heston_paths, _ = heston_data_generator(
            **_params_with_path_count(heston_params, n_heston, seed)
        )
        parts.append(heston_paths)

    if n_garch > 0:
        garch_paths, _ = garch_data_generator(
            **_params_with_path_count(garch_params, n_garch, seed + 1)
        )
        parts.append(garch_paths)

    row_counts = {part.shape[0] for part in parts}
    if len(row_counts) != 1:
        raise ValueError("Heston and GARCH paths must have the same number of time steps.")

    mixed_paths = pd.concat(parts, axis=1, ignore_index=True)
    permutation = np.random.default_rng(seed + 2).permutation(mixed_paths.shape[1])
    mixed_paths = mixed_paths.iloc[:, permutation]
    mixed_paths.columns = range(mixed_paths.shape[1])

    return mixed_paths, n_heston, n_garch


def _set_torch_seed(seed, device):
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))


def _train_mix_model(train_main, model_class, model_kwargs, train_loader,
                     val_loader, option_type_fn, strike, cost_rate, lam,
                     alpha, epochs, learning_rate, use_aadh,
                     adaptive_learning, schedule, device, seed, quiet):
    _set_torch_seed(seed, device)
    model = model_class(**model_kwargs)
    call = lambda: train_main(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        option_type_fn=option_type_fn,
        strike=strike,
        cost_rate=cost_rate,
        lam=lam,
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        AADH=use_aadh,
        adaptive_learning=adaptive_learning,
        schedule=schedule,
        device=device
    )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return call()
    return call()


def run_mixed_generator_experiment(heston_params,
                                   garch_params,
                                   n_train_paths=2000,
                                   n_test_paths=1000,
                                   xi_step=0.1,
                                   xi_values=None,
                                   n_clusters=5,
                                   batch_size=100,
                                   train_val_proportions=(0.8, 0.2),
                                   epochs=50,
                                   cost_rate=0.0,
                                   learning_rate=1e-3,
                                   lam=1/100,
                                   alpha=250,
                                   option_type_fn=None,
                                   strike=1.0,
                                   adaptive_learning=False,
                                   schedule=None,
                                   model_class=None,
                                   model_kwargs=None,
                                   device="cpu",
                                   seed=42,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=None,
                                   persistent_workers=False,
                                   quiet=True):
    """
    Train AADH and SDH on Heston/GARCH mixtures and test on all mixtures.

    xi is the Heston weight in the mixture:
        xi = 0.0 -> 0% Heston, 100% GARCH
        xi = 1.0 -> 100% Heston, 0% GARCH

    Returns a dataframe with one row per train/test mixture pair.
    """
    import torch
    from torch.utils.data import DataLoader

    from ambiguity_measures.KMeans_algo import compute_feature_matrix, kmeans_clustering
    from dataset.dataset import AADeepHedgingDataset, loader as build_loaders
    from models.neural_network import HedgingNN
    from risk_measures.test_metrics import test_result_metrics
    from training.train_loop_nn import main as train_main
    from utils.options import eur_call_payoff

    if n_train_paths <= 0 or n_test_paths <= 0:
        raise ValueError("n_train_paths and n_test_paths must be positive.")

    if len(train_val_proportions) != 2:
        raise ValueError("train_val_proportions must contain train and validation proportions.")

    train_prop, val_prop = train_val_proportions
    if abs((train_prop + val_prop) - 1.0) > 1e-10:
        raise ValueError("train_val_proportions must sum to 1.")

    if model_class is None:
        model_class = HedgingNN
    if model_kwargs is None:
        model_kwargs = {}
    if option_type_fn is None:
        option_type_fn = eur_call_payoff

    device = torch.device(device)
    xi_grid = _mixture_grid(xi_step=xi_step, xi_values=xi_values)
    test_path_cache = {}

    for test_idx, test_xi in enumerate(xi_grid):
        paths, n_heston, n_garch = _generate_mixed_paths(
            heston_params,
            garch_params,
            n_test_paths,
            test_xi,
            seed + 100_000 + test_idx * 1_000
        )
        test_path_cache[test_xi] = {
            "paths": paths,
            "n_heston": n_heston,
            "n_garch": n_garch,
        }

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }

    rows = []

    for train_idx, train_xi in enumerate(xi_grid):
        train_paths, train_heston_count, train_garch_count = _generate_mixed_paths(
            heston_params,
            garch_params,
            n_train_paths,
            train_xi,
            seed + 10_000 + train_idx * 1_000
        )

        train_features = compute_feature_matrix(train_paths)
        train_labels, kmeans_model = kmeans_clustering(train_features, n_clusters=n_clusters)
        train_dataset = AADeepHedgingDataset(train_paths, train_labels)
        train_loader, val_loader, _ = build_loaders(
            train_dataset,
            batch_size=batch_size,
            set_proportions=[train_prop, val_prop, 0.0],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        model_seed = seed + train_idx
        aadh_model = _train_mix_model(
            train_main, model_class, model_kwargs, train_loader, val_loader,
            option_type_fn, strike, cost_rate, lam, alpha, epochs,
            learning_rate, True, adaptive_learning, schedule, device,
            model_seed, quiet
        )
        sdh_model = _train_mix_model(
            train_main, model_class, model_kwargs, train_loader, val_loader,
            option_type_fn, strike, cost_rate, lam, alpha, epochs,
            learning_rate, False, adaptive_learning, schedule, device,
            model_seed, quiet
        )

        for test_xi in xi_grid:
            test_paths = test_path_cache[test_xi]["paths"]
            test_features = compute_feature_matrix(test_paths)
            test_labels = kmeans_model.predict(test_features)
            test_dataset = AADeepHedgingDataset(test_paths, test_labels)
            test_loader = DataLoader(test_dataset, **loader_kwargs)

            aadh_risk, aadh_variance, aadh_costs = test_result_metrics(
                aadh_model,
                test_loader,
                option_type_fn=option_type_fn,
                strike=strike,
                cost_rate=cost_rate,
                lam=lam,
                device=device
            )
            sdh_risk, sdh_variance, sdh_costs = test_result_metrics(
                sdh_model,
                test_loader,
                option_type_fn=option_type_fn,
                strike=strike,
                cost_rate=cost_rate,
                lam=lam,
                device=device
            )

            rows.append({
                "train_xi_heston": train_xi,
                "test_xi_heston": test_xi,
                "train_heston_paths": train_heston_count,
                "train_garch_paths": train_garch_count,
                "test_heston_paths": test_path_cache[test_xi]["n_heston"],
                "test_garch_paths": test_path_cache[test_xi]["n_garch"],
                "AADH_overall_risk": aadh_risk,
                "AADH_variance": aadh_variance,
                "AADH_transaction_costs": aadh_costs,
                "SDH_overall_risk": sdh_risk,
                "SDH_variance": sdh_variance,
                "SDH_transaction_costs": sdh_costs,
                "risk_diff_SDH_minus_AADH": sdh_risk - aadh_risk,
                "variance_diff_SDH_minus_AADH": sdh_variance - aadh_variance,
                "cost_diff_SDH_minus_AADH": sdh_costs - aadh_costs,
            })

    return pd.DataFrame(rows)




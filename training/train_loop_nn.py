"""
Training loop for the neural network hedging model.
We define functions to train and validate the model for one epoch,
as well as the main training loop that iterates over multiple epochs.

The functions defined are:
1. train_one_epoch_AADH: trains the model for one epoch using the AADH approach with ambiguity measures.
2. validate_AADH: validates the model for one epoch using the AADH approach
3. train_one_epoch_SDH: trains the model for one epoch using the standard deep hedging approach.
4. validate_SDH: validates the model for one epoch using the standard deep hedging approach.
5. main: the main training loop that orchestrates the training and validation
    over multiple epochs, with options for adaptive learning rate and choice
    between AADH and SDH approaches.

In the main file we only import the main function.

Note: In the functions we intentionally left some print statements commented out,
        to allow for an easier monitoring of the training process.

"""
# Adjust path for imports
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# import libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
# Import modules from other scripts
from utils.options import *
from utils.PL import comp_PL_components
from risk_measures.measures import AADHInnerLoss
from ambiguity_measures.ambiguity_loss_fn import AADHOuterLoss
from models.neural_network import *
# Import train_utils functions
from training.train_utils import adjust_learning_rate, calculate_cluster_probabilities_loader


def _to_device(tensor, device):
    return tensor.to(device, non_blocking=True)


def _cluster_probs_to_tensor(cluster_probs, device):
    """
    Convert the cluster probability dictionary to an indexable tensor once.
    The tensor index is the cluster label.
    """
    if torch.is_tensor(cluster_probs):
        return cluster_probs.to(device=device, dtype=torch.float32)

    if not cluster_probs:
        return torch.empty(0, dtype=torch.float32, device=device)

    max_label = max(int(label) for label in cluster_probs.keys())
    cluster_probs_tensor = torch.zeros(max_label + 1, dtype=torch.float32, device=device)

    for label, info in cluster_probs.items():
        cluster_probs_tensor[int(label)] = float(info["probability"])

    return cluster_probs_tensor


def _cluster_probs_for_batch(cluster_probs_tensor, unique_labels):
    cluster_probs_batch = cluster_probs_tensor[unique_labels]
    return cluster_probs_batch / cluster_probs_batch.sum()


def _can_use_full_batch_aadh_forward(model):
    """
    Full-batch AADH forwarding is equivalent for the current feed-forward model.
    If future models add training-time batch-dependent layers, fall back to the
    old per-cluster forward behavior.
    """
    batch_dependent_layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    return not any(isinstance(module, batch_dependent_layers) for module in model.modules())


def _compute_aadh_batch_loss(model, paths, labels, cluster_probs_tensor,
                             inner_loss, outer_loss, option_type_fn,
                             strike, cost_rate, use_full_batch_forward=True):
    inner_losses = []
    unique_labels = labels.unique()

    if use_full_batch_forward:
        delta = model(paths)
        PL_batch, TC_batch, _ = comp_PL_components(delta,
                                                   paths,
                                                   payoff_fn=option_type_fn,
                                                   payoff_kwargs={'strike': strike},
                                                   cost_rate=cost_rate,
                                                   p0=0.0)

        for label in unique_labels:
            mask = (labels == label)
            inner_losses.append(inner_loss(PL_batch[mask], TC_batch[mask]))
    else:
        for label in unique_labels:
            mask = (labels == label)
            paths_cluster = paths[mask]

            delta = model(paths_cluster)
            PL_cluster, TC_cluster, _ = comp_PL_components(delta,
                                                           paths_cluster,
                                                           payoff_fn=option_type_fn,
                                                           payoff_kwargs={'strike': strike},
                                                           cost_rate=cost_rate,
                                                           p0=0.0)
            inner_losses.append(inner_loss(PL_cluster, TC_cluster))

    cluster_probs_batch = _cluster_probs_for_batch(cluster_probs_tensor, unique_labels)
    inner_losses_tensor = torch.stack(inner_losses)
    return outer_loss(inner_losses_tensor, cluster_probs_batch)


# Function to train one epoch AADH
def train_one_epoch_AADH(model, loader, optimizer, cluster_probs, inner_loss=AADHInnerLoss(), 
                    outer_loss=AADHOuterLoss(), option_type_fn=eur_call_payoff, 
                    strike=100.0, cost_rate=0.0, device='cpu'):
    """
    With this function we train one epoch of the neural network model.
    We loop through the data loader for each batch and we loop through the clusters
    to compute the inner and outer loss functions.
    We use a mask to separate the different clusters and compute the inner
    loss for each cluster, then we compute the outer loss on the inner losses.
    Finally we backpropagate the loss and update the model parameters.

    """
    model.train()
    device = torch.device(device)
    cluster_probs_tensor = _cluster_probs_to_tensor(cluster_probs, device)
    use_full_batch_forward = _can_use_full_batch_aadh_forward(model)
    running_loss = torch.zeros((), device=device)
    n = 0
    # iterate through the data loader to get batches
    for paths, labels in loader:
        paths = _to_device(paths, device) # shape (n_paths_in_batch, n_steps+1, 1)
        labels = _to_device(labels, device) # shape (n_paths_in_batch,)
        batch_size = paths.size(0)

        """ Info prints """
        # print("Cluster probabilities batch:", cluster_probs_batch)
        # print(inner_losses)

        outer_loss_value = _compute_aadh_batch_loss(model, paths, labels,
                                                    cluster_probs_tensor,
                                                    inner_loss, outer_loss,
                                                    option_type_fn, strike,
                                                    cost_rate,
                                                    use_full_batch_forward)
        # backpropagate the loss and update the model parameters
        optimizer.zero_grad(set_to_none=True) # clear previous gradients
        outer_loss_value.backward()
        clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
        optimizer.step() # update model parameters
        # accumulate running loss
        running_loss += outer_loss_value.detach() * batch_size
        n += batch_size

    epoch_loss = (running_loss / n).item()

    """ Info prints """
    # print(outer_loss_value)
    # print("Epoch training loss:", epoch_loss)
    
    return epoch_loss


# Validation function AADH
def validate_AADH(model, loader, cluster_probs, inner_loss=AADHInnerLoss(), 
             outer_loss=AADHOuterLoss(), option_type_fn=eur_call_payoff, 
             strike=100.0, cost_rate=0.0, device='cpu'):
    """
    For the validation function, we follow the same structure as in train_one_epoch,
    but we do not backpropagate the loss and we do not update the model parameters.
    We also set the model to evaluation mode and disable gradient computation.

    """
    model.eval()
    device = torch.device(device)
    cluster_probs_tensor = _cluster_probs_to_tensor(cluster_probs, device)
    running_loss = torch.zeros((), device=device)
    n = 0
    with torch.inference_mode():
        for paths, labels in loader:
            paths = _to_device(paths, device)
            labels = _to_device(labels, device)
            batch_size = paths.size(0)

            """ Info prints """
            # print("Cluster probabilities batch (val):", cluster_probs_batch)
            # print(inner_losses)

            outer_loss_value = _compute_aadh_batch_loss(model, paths, labels,
                                                        cluster_probs_tensor,
                                                        inner_loss, outer_loss,
                                                        option_type_fn, strike,
                                                        cost_rate,
                                                        use_full_batch_forward=True)

            running_loss += outer_loss_value * batch_size
            n += batch_size

    epoch_loss = (running_loss / n).item()

    """ Info prints """
    # print("Epoch validation loss:", epoch_loss)

    return epoch_loss


# Train one epoch function SDH
def train_one_epoch_SDH(model, loader, optimizer, loss=AADHInnerLoss(), 
                        option_type_fn=eur_call_payoff, strike=100.0, 
                        cost_rate=0.0, device='cpu'):
    """
    We use a similar structure as in train_one_epoch_AADH, but we compute only the
    inner loss for all paths in the batch, without separating them into clusters.

    """
    model.train()
    device = torch.device(device)
    running_loss = torch.zeros((), device=device)
    n = 0
    # iterate through the data loader to get batches
    for paths, _ in loader:
        paths = _to_device(paths, device) # shape (n_paths_in_batch, n_steps+1, 1)
        batch_size = paths.size(0)

        # compute the delta using the model
        delta = model(paths) # shape (batch_size, n_steps, 1)
        PL_batch, TC_batch, _ = comp_PL_components(delta,
                                                   paths,
                                                   payoff_fn=option_type_fn,
                                                   payoff_kwargs={'strike': strike},
                                                   cost_rate=cost_rate,
                                                   p0=0.0)
        # compute the loss for the batch
        loss_value = loss(PL_batch, TC_batch)  # scalar tensor
        # backpropagate the loss and update the model parameters
        optimizer.zero_grad(set_to_none=True) # clear previous gradients
        loss_value.backward()
        clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
        optimizer.step() # update model parameters
        # accumulate running loss
        running_loss += loss_value.detach() * batch_size
        n += batch_size
    epoch_loss = (running_loss / n).item()

    """ Info prints """
    # print("Epoch training loss (SDH):", epoch_loss)
    
    return epoch_loss


# Validation function SDH
def validate_SDH(model, loader, inner_loss=AADHInnerLoss(),
                  option_type_fn=eur_call_payoff, strike=100.0, 
                  cost_rate=0.0, device='cpu'):
    """
    Similar to the train_one_epoch_SDH function, but without backpropagation
    and model parameter updates.

    """
    model.eval()
    device = torch.device(device)
    running_loss = torch.zeros((), device=device)
    n = 0
    with torch.inference_mode():
        for paths, _ in loader:
            paths = _to_device(paths, device)
            batch_size = paths.size(0)

            delta = model(paths)

            PL_cluster, TC_cluster, _ = comp_PL_components(delta,
                                                           paths,
                                                           payoff_fn=option_type_fn,
                                                           payoff_kwargs={'strike': strike},
                                                           cost_rate=cost_rate,
                                                           p0=0.0)
            
            loss_value = inner_loss(PL_cluster, TC_cluster)

            running_loss += loss_value * batch_size
            n += batch_size
    epoch_loss = (running_loss / n).item()

    """ Info prints """
    # print("Epoch validation loss (SDH):", epoch_loss)
    return epoch_loss


# Main training loop function
def main(model, train_loader, val_loader,
         option_type_fn=eur_call_payoff, strike=100.0, cost_rate=0.0, 
         lam=1/100, alpha=250, epochs=100, learning_rate=1e-3, AADH=True,
         adaptive_learning=False, schedule=None, 
         device='cpu'):
    """
    We create a main learning loop for the neural networ. 
    Depending on the AADH flag, we use either the AADH training and validation functions
    or the SDH ones.
    With adaptive learning, we reduce the learning rate periodically as epochs progress.

    """
    # Calculate automatically cluster probabilities from train_loader
    cluster_probs = calculate_cluster_probabilities_loader(train_loader)
    # Move model to device
    device = torch.device(device)
    model.to(device)
    if AADH:
        cluster_probs = _cluster_probs_to_tensor(cluster_probs, device)
    # Define optimizers
    optimizer_AADH = Adam(model.parameters(), lr=learning_rate) if AADH else None
    optimizer_SDH = Adam(model.parameters(), lr=learning_rate) if not AADH else None
    inner_loss_fn = AADHInnerLoss(lam=lam)
    outer_loss_fn = AADHOuterLoss(alpha=alpha)
    # Training loop
    # If condition for adaptive learning rate
    if adaptive_learning:
        # if condition for AADH
        if AADH:
            for epoch in range(1, epochs+1):
                adjust_learning_rate(optimizer_AADH, epoch=epoch, starting_lr=learning_rate,
                                     schedule=schedule)
                # print epoch and current learning rate
                # We take just the learning rate of the first parameter group, since it is the same for all
                print(f"Epoch {epoch}/{epochs} - Learning Rate: {optimizer_AADH.param_groups[0]['lr']}")
                train_loss = train_one_epoch_AADH(model, train_loader, optimizer_AADH, cluster_probs,
                                                 inner_loss=inner_loss_fn,
                                                 outer_loss=outer_loss_fn,
                                                 option_type_fn=option_type_fn,
                                                 strike=strike,
                                                 cost_rate=cost_rate,
                                                 device=device)
                val_loss = validate_AADH(model, val_loader, cluster_probs,
                                         inner_loss=inner_loss_fn,
                                         outer_loss=outer_loss_fn,
                                         option_type_fn=option_type_fn,
                                         strike=strike,
                                         cost_rate=cost_rate,
                                         device=device)
                print(f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}\n")
                # print dashed line for better readability
                print("-" * 55)
        else:
            for epoch in range(1, epochs+1):
                adjust_learning_rate(optimizer_SDH, epoch, starting_lr=learning_rate,
                                     schedule=schedule)
                print(f"Epoch {epoch}/{epochs} - Learning Rate: {optimizer_SDH.param_groups[0]['lr']}")
                train_loss = train_one_epoch_SDH(model, train_loader, optimizer_SDH,
                                                loss=inner_loss_fn,
                                                option_type_fn=option_type_fn,
                                                strike=strike,
                                                cost_rate=cost_rate,
                                                device=device)
                val_loss = validate_SDH(model, val_loader,
                                        inner_loss=inner_loss_fn,
                                        option_type_fn=option_type_fn,
                                        strike=strike,
                                        cost_rate=cost_rate,
                                        device=device)
                print(f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}\n")
                print("-" * 55)
    else:
        # if condition for AADH
        if AADH:
            for epoch in range(1, epochs+1):
                print(f"Epoch {epoch}/{epochs}")
                train_loss = train_one_epoch_AADH(model, train_loader, optimizer_AADH, cluster_probs,
                                                 inner_loss=inner_loss_fn,
                                                 outer_loss=outer_loss_fn,
                                                 option_type_fn=option_type_fn,
                                                 strike=strike,
                                                 cost_rate=cost_rate,
                                                 device=device)
                val_loss = validate_AADH(model, val_loader, cluster_probs,
                                         inner_loss=inner_loss_fn,
                                         outer_loss=outer_loss_fn,
                                         option_type_fn=option_type_fn,
                                         strike=strike,
                                         cost_rate=cost_rate,
                                         device=device)
                print(f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}\n")
                print("-" * 55)
        else:
            for epoch in range(1, epochs+1):
                print(f"Epoch {epoch}/{epochs}")
                train_loss = train_one_epoch_SDH(model, train_loader, optimizer_SDH,
                                                loss=inner_loss_fn,
                                                option_type_fn=option_type_fn,
                                                strike=strike,
                                                cost_rate=cost_rate,
                                                device=device)
                val_loss = validate_SDH(model, val_loader,
                                        inner_loss=inner_loss_fn,
                                        option_type_fn=option_type_fn,
                                        strike=strike,
                                        cost_rate=cost_rate,
                                        device=device)
                print(f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}\n")
                print("-" * 55)

    return model


"""
This script contains the neural network model with configurable layers and neurons.

We create two modules, one following the original DH paper (semirecurrent), the other following
the AADH one (classical MLP).
Preliminary tests shows that the semirecurrent beats the MLP on synthetic data.

The output has no activation.

(Later added an activation also in the output layer to test with different lambdas)

We generalize the function with n_assets and n_features as inputs in order to be 
able to use it in future extensions.
In our case n_assets=1 (or d in the paper DH) and n_features=2 (price and previous delta).

In the paper "Ambiguity-Averse Deep Hedging with Feature Clustering" by Jones et al., 2025,
the NN inputs are current prices, current deltas and the time to maturity (T-t).

In further implementations, the network can be easily extended:

1. To include more features (e.g., volatility) by changing the n_features parameter.
    T-t can be added inside the forward def without changing the data loader and the forward loop structure,
    by adding as input the TTM and using the length of the S_seq used as input.
    
    Note well: use for TTM the proper scale, e.g. 30/252 - 10/252 for an option that matures in
    20 days. 

2. To include more assets by changing the n_assets parameter.

3. To change the architecture (e.g., adding dropout, batch norm, different activations, etc.)

"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import utils
from models.models_utils import CustomSharpSigmoid

# Create class for the semi-recurrent neural network model
class HedgingNN(nn.Module):
    """
    We use four imputs:
    The number of hidden layers, the number of neurons per hidden layer, 
    number of features and number of assets

    As for the NN, the input is given by the number of assets d multiplied 
    by the the number of features.
    The output is given by the number of assets d (in our case d=1)
    """
    def __init__(self, 
                 hidden_layers_dim=100, 
                 num_hidden_layers=3,
                 # these two below are "locked" for now
                 # added in case of future extensions
                 n_features=3, # remember to set n_features=2 for price and previous delta, 3 for time to matruity
                 n_assets=1,
                 center=0.5,
                 scale=10.0):
        super(HedgingNN, self).__init__()
        # input_dim is given by the number of assets d multiplied by the the number of features
        # in our case d=1 and we have the price and the previous delta as features, so input_dim=2

        self.hidden_layers_dim = hidden_layers_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_assets = n_assets
        self.n_features = n_features
        # Define input dimension
        input_dim = n_assets * n_features  # in our case 1*2=2
        # Define the layers
        layers = []
        # input layer, with nn.Linear(input_dim, output_dim)
        layers.append(nn.Linear(input_dim, hidden_layers_dim))
        layers.append(nn.ReLU())

        for _ in range(self.num_hidden_layers):
            # hidden layers with nn.Linear(input_dim, output_dim)
            layers.append(nn.Linear(hidden_layers_dim, hidden_layers_dim))
            layers.append(nn.ReLU())

        # output layer with nn.Linear(input_dim, output_dim)
        # The output dim is given by the number of assets d (in our case d=1)
        layers.append(nn.Linear(hidden_layers_dim, n_assets))
        
        """
        Here we added a sigmoid activation at the output layer to constrain the deltas
        between 0 and 1 for a call option.
        """
        # Constrain output of deltas to be between 0 and 1 for a call option
        # layers.append(nn.Sigmoid())
        layers.append(CustomSharpSigmoid(center=center, scale=scale))

        """
        Here we added a custom sharp sigmoid activation at the output layer to constrain the deltas
        between 0 and 1 for a call option. The difference is the scale and center factor to 
        reflect the normal learning dynamics of the NN.
        """

        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)


    def forward(self, S_seq):
        """
        Run forward pass of the network
        Inputs:
        S_seq: tensor of shape [batch_size, n_steps + 1, d]
        delta_prev: tensor of shape [batch_size, d]
        Returns:
        delta: tensor of shape [batch_size, n_steps, d]
        For just one asset d=1
        """
        batch_size, n_steps_plus_1, d = S_seq.shape
        n_steps = n_steps_plus_1 - 1
        assert d == self.n_assets, "Number of assets does not match"

        delta_seq = [None] * n_steps
        delta_prev = torch.zeros(batch_size, d, device=S_seq.device, dtype=S_seq.dtype)
        time_to_maturity_values = (
            torch.arange(n_steps, 0, -1, device=S_seq.device, dtype=S_seq.dtype)
            .view(n_steps, 1)
            / 252
        )

        for k in range(n_steps):
            S_k = S_seq[:,k,:] # [B,d]
            time_to_maturity_k = time_to_maturity_values[k].expand(batch_size, 1)
            input = torch.cat([S_k, delta_prev, time_to_maturity_k], dim=-1) # [B,2d]
            delta_k = self.network(input) # [B,d]
            delta_seq[k] = delta_k
            delta_prev = delta_k

        delta = torch.stack(delta_seq, dim=1)   # shape [batch_size, n_steps, d]
        return delta





"""
We create a class for a neural network model that follows the approach of
"Ambiguity-Averse Deep Hedging with Feature Clustering" by Jones et al., 2025.

In this model, the NN inputs are the current prices, the current deltas and the time to maturity (T-t).
As output we have the deltas at the next time step.
"""

class Hedging_FF_NN_TTM_Feature(nn.Module):
    def __init__(self, 
                 hidden_layers_dim=100, 
                 num_hidden_layers=3,
                 # these two below are "locked" for now
                 # added in case of future extensions
                 n_features=3, # remember to set n_features=3 for price, delta and time to maturity
                 n_assets=1,
                 center=0.5,
                 scale=10.0):
        super(Hedging_FF_NN_TTM_Feature, self).__init__()
        # input_dim is given by the number of assets d multiplied by the the number of features
        # in our case d=1 and we have the price and the previous delta as features, so input_dim=2

        self.hidden_layers_dim = hidden_layers_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_assets = n_assets
        self.n_features = n_features
        # Define input dimension
        input_dim = n_assets * n_features  # in our case 1*3=3
        # Define the layers
        layers = []
        # input layer, with nn.Linear(input_dim, output_dim)
        layers.append(nn.Linear(input_dim, hidden_layers_dim))
        layers.append(nn.ReLU())

        for _ in range(self.num_hidden_layers):
            # hidden layers with nn.Linear(input_dim, output_dim)
            layers.append(nn.Linear(hidden_layers_dim, hidden_layers_dim))
            layers.append(nn.ReLU())

        # output layer with nn.Linear(input_dim, output_dim)
        # The output dim is given by the number of assets d (in our case d=1)
        layers.append(nn.Linear(hidden_layers_dim, n_assets))

        # added sigmoid activation at the output layer to constrain the deltas
        layers.append(CustomSharpSigmoid(center=center, scale=scale))

        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, S_seq):
        """
        Run forward pass of the network
        Inputs:
        S_seq: tensor of shape [batch_size, n_steps + 1, d]
        delta_k: tensor of shape [batch_size, d]
        Returns:
        delta: tensor of shape [batch_size, n_steps, d]
        For just one asset d=1

        We normalize time to maturity as (T-t)/252 to have it in years.
        If needed, the scaling factor can be changed, but cannot be done outside this function.

        """
        batch_size, n_steps_plus_1, d = S_seq.shape
        n_steps = n_steps_plus_1 - 1
        assert d == self.n_assets, "Number of assets does not match"

        delta_seq = [None] * n_steps
        delta_k = torch.zeros(batch_size, d, device=S_seq.device, dtype=S_seq.dtype)
        time_to_maturity_values = (
            torch.arange(n_steps, 0, -1, device=S_seq.device, dtype=S_seq.dtype)
            .view(n_steps, 1)
            / 252
        )

        for k in range(n_steps):
            S_k = S_seq[:,k,:] # [B,d]
            time_to_maturity_k = time_to_maturity_values[k].expand(batch_size, 1)
            input = torch.cat([S_k, delta_k, time_to_maturity_k], dim=-1) # [B,3d]
            delta_k_plus_one = self.network(input) # [B,d]
            delta_seq[k] = delta_k_plus_one
            delta_k = delta_k_plus_one
        delta = torch.stack(delta_seq, dim=1)   # shape [batch_size, n_steps, d]
        return delta



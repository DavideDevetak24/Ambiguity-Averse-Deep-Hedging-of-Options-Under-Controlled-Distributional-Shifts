# Ambiguity-Averse Deep Hedging for Options Under Controlled Distributional Shifts Experiments

This repository contains an exploratory implementation around Standard Deep
Hedging (SDH) and Ambiguity-Averse Deep Hedging (AADH) with feature clustering and is part of the master thesis "Ambiguity Averse Deep Hedging of Options Under Controlled Distributional Shifts".
The code was developed starting from the original deep hedging/AADH ideas and
then extended with several additional checks. We explored many setups of the underlying models, however, in this repository we only report the main findings and main experiments.

## Library Structure

- `data_generators/`: synthetic path generators.
  - `heston.py`: Heston price/variance paths and option-price utilities.
  - `garch.py`: GARCH(1,1) paths used for distributional-shift tests.
  - `cev.py`: CEV paths used as an additional out-of-model test set.
- `dataset/`: PyTorch dataset and dataloader helpers.
  - `dataset.py`: wraps path data and clustering labels.
  - `real_data/`: empirical index data and scripts for sliding-window datasets.
- `models/`: neural-network hedgers.
  - `neural_network.py`: semi-recurrent and feedforward hedging networks.
  - `models_utils.py`: output activation utilities, including a sharp sigmoid
    used in some constrained-output experiments.
- `utils/`: payoff, P&L, plotting, and chart helpers.
  - `PL.py`: core self-financing P&L and transaction-cost convention without
    forced final liquidation.
  - `options.py`: European call/put, lookback call, and Asian call payoffs.
- `risk_measures/`: inner risk/loss definitions and evaluation metrics.
- `ambiguity_measures/`: KMeans feature clustering and the AADH outer loss.
- `training/`: SDH/AADH training loops, learning-rate utilities, and mixed
  Heston/GARCH experiment helpers.

## Experiment Notebooks

- `Main.ipynb`: baseline workflow. It generates Heston training paths, applies
  KMeans feature clustering, trains SDH and AADH, and evaluates on held-out
  Heston, Heston/GARCH mixtures, and empirical data.
- `main_mix.ipynb`: Heston/GARCH
  mixture experiments. This notebook study distributional shifts by changing
  the mixture weight between Heston-like and GARCH-like paths at training and/or
  testing time.

## Notes For Readers

- In the experiments, the conventions follow the original Deep Hedging paper
  by Buehler et al. The semi-recurrent model (variant with a single neural network with time as input) is kept as an important reference because it generally gave better performance than the MLP variants (these were tested but are not present in this repository).
- Empirical checks use normalized sliding windows from Open and Close prices.
  Some notebooks compare the correct chronological window order with the
  opposite order as a diagnostic.

## Main References
- Jones, A., Horvath, B., Reisinger, C., Wood, B., Bai, L., & Akkari, A. (2025).
Ambiguity-averse deep hedging with feature clustering. Available at SSRN 5390563
- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. Quanti-
tative Finance, 19 (8), 1271–1291



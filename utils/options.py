"""
In this script we define the options that can be used in the main script
Every def takes a price path and computes the value of the option at time T
The data here is already in tensor form, as constructed in dataset/dataset.py

Options defined:
1. European Call
2. European Put
3. Lookback Call
4. Asian Call
"""
import torch

def eur_call_payoff(S_seq, strike=100):
    S_T = S_seq[:, -1, 0]  # final price
    return torch.clamp(S_T - float(strike), min=0.0)

def eur_put_payoff(S_seq, strike=100):
    S_T = S_seq[:, -1, 0]
    return torch.clamp(float(strike) - S_T, min=0.0)

# lookback call option
def lookback_call_payoff(S_seq, strike=100):
    S_max = S_seq.max(dim=1).values[:,0]  # max over time, return batch size
    return torch.clamp(S_max - float(strike), min=0.0)

# asian call option
def asian_call_payoff(S_seq, strike=100):
    # mean over 252 (dim 1)
    S_avg = S_seq.mean(dim=1)[:,0]
    return torch.clamp(S_avg - float(strike), min=0.0)




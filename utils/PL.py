"""
In this script we define the hedging P&L computation functions
We use the original version proposed in the Deep Hedging paper by Buehler et al.
and we extend it for the ambiguity-averse case by separating P&L before 
transaction costs and transaction costs.

In practice we compute P&L as PL(Z, delta, p0) = -Z + p0 + (delta * S) - C(delta)

After that PL_before_costs(Z, delta) = -Z + p0 + (delta * S)
and transaction costs as C(delta) = PL_before_costs - PL
The last two will be used in the inner loss funcction

The tensors are expected to be of shape:
- S_seq: (batch_size, n_steps + 1, 1)
- delta: (batch_size, n_steps, 1)
- Z: (batch_size,)
- costs: (batch_size,)

We also import the options defined in options.py to be used as payoff functions
The options will be used dynamically by passing the function handle to the PL functions

To summarize, we define the following functions:
1. comp_delta_S: computes the dot product of delta and S_seq over time steps
2. comp_costs: computes the transaction costs given delta and S_seq
3. comp_PL: computes the P&L in two versions, with and without transaction costs

In the experiments p0 is generally set to zero. For a convex inner risk functional, it
might be useful to calculate the indifference price.

"""

# import libraries
import torch

def comp_delta_S(delta, S_seq):
    """
    Compute the dot product of delta and S_seq over time steps
    We do in tensor form the following calculation:

    sum_{t=0}^{T-1} delta_t * (S_{t+1} - S_t)

    with delta of shape [batch, T, d] and S_seq of shape [batch, T+1, d]

    """
    S_diff = S_seq[:, 1:, :] - S_seq[:, :-1, :]  # [batch, T, d]
    delta_dot_s = (delta * S_diff).sum(dim=(1, 2))  # [batch], sum over dim 1 and 2 (T and d)
    return delta_dot_s

def comp_costs(delta, S_seq, cost_rate=0.0):
    """
    The function computes the costs doing the following calculation:
    
    sum_{t=0}^{T-1} |delta_t - delta_{t-1}| * S_t * cost_rate

    Delta has a shape of [batch, T, d] but it has to be enlarged to [batch, T+1, d]
    since delta_{-1} = 0 (otherwise the first trade cannot be computed)
    S_seq has shape [batch, T+1, d], but we only need S_0..S_{T-1} for costs calculation
    so we use S_seq[:, :-1, :]

    """
    # Cut the seq to get S_0..S_{T-1}
    S_seq_truncated = S_seq[:, :-1, :]  # [batch, T, d]
    # Compare delta_t against delta_{t-1}, with delta_{-1}=0.
    delta_prev = torch.cat([torch.zeros_like(delta[:, :1, :]), delta[:, :-1, :]], dim=1)
    trades = delta - delta_prev  # [batch, T, d]
    abs_trades = trades.abs()

    costs = (abs_trades * S_seq_truncated * cost_rate).sum(dim=(1, 2))  # [batch]
    return costs


def comp_PL_components(delta, S_seq, payoff_fn=None, payoff_kwargs=None,
                       cost_rate=0.0, p0=0.0):
    """
    Compute P&L before costs, transaction costs, and P&L after costs together.
    This avoids recomputing payoff and delta*S when all components are needed.
    """
    batch = S_seq.shape[0]
    device = S_seq.device
    dtype = S_seq.dtype

    if payoff_fn is None:
        Z = torch.zeros(batch, device=device, dtype=dtype)
    else:
        Z = payoff_fn(S_seq, **(payoff_kwargs or {}))
        Z = Z.type_as(S_seq)

    delta_s = comp_delta_S(delta, S_seq)
    costs = comp_costs(delta, S_seq, cost_rate=cost_rate)
    pl_before_costs = -Z + float(p0) + delta_s
    pl_after_costs = pl_before_costs - costs

    return pl_before_costs, costs, pl_after_costs


def comp_PL(delta, S_seq, payoff_fn=None, payoff_kwargs=None, 
            cost_rate=0.0, p0=0.0, transaction_costs=False):
    """
    Compute the P&L as defined in the formula above
    We add a transaction_costs boolean to choose whether to compute P&L
    with or without transaction costs. The first case is used for the classic
    deep hedging, while the second case is used for the inner loss function
    in the ambiguity-averse deep hedging.    

    """
    if transaction_costs:
        _, _, pl = comp_PL_components(delta, S_seq,
                                      payoff_fn=payoff_fn,
                                      payoff_kwargs=payoff_kwargs,
                                      cost_rate=cost_rate,
                                      p0=p0)
        return pl

    batch = S_seq.shape[0]
    device = S_seq.device
    dtype = S_seq.dtype

    if payoff_fn is None:
        Z = torch.zeros(batch, device=device, dtype=dtype)
    else:
        Z = payoff_fn(S_seq, **(payoff_kwargs or {}))  # full path
        Z = Z.type_as(S_seq) # shape [batch]

    delta_s = comp_delta_S(delta, S_seq)
    pl = -Z + float(p0) + delta_s

    return pl



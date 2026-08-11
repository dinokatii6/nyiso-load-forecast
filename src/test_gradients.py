""" 
Verifies that the gradients in the MLP pipeline are correct. 
Two independent checks are used: 
1. Finite differences compare the analytical gradients from backward() against numerical gradients computed using only forward() and loss(). 
2. PyTorch autograd computes an independent reference gradient using the same network architecture and copied weights. 
"""
from __future__ import annotations

import numpy as np

from src.full_mlp import TwoLayerMLP

RTOL_PASS = 1e-6


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a) + abs(b), 1e-8)


def make_toy(n=17, d=5, h=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=(n, 1))
    return X, y, TwoLayerMLP(d, h, seed=seed)


# Check 1: finite differences
def check_finite_differences(eps=1e-5, n_samples=25, seed=0):
    X, y, net = make_toy(seed=seed)

    analytic = net.backward(net.forward(X), y)
    params = net.get_params()
    rng = np.random.default_rng(123)

    worst = 0.0
    print("\n--- finite-difference check ---")

    for name, param in params.items():
        flat = param.ravel()               
        analytic_flat = analytic[name].ravel()

        # A random sample catches any systematic error.
        picks = rng.choice(flat.size, size=min(n_samples, flat.size), replace=False)

        for k in picks:
            original = flat[k]

            flat[k] = original + eps
            loss_plus = net.loss(net.forward(X), y)

            flat[k] = original - eps
            loss_minus = net.loss(net.forward(X), y)

            flat[k] = original          # restore BEFORE moving on

            numeric = (loss_plus - loss_minus) / (2 * eps)
            err = relative_error(numeric, analytic_flat[k])
            worst = max(worst, err)

        print(f"  {name:>3}  worst relative error so far: {worst:.3e}")

    print(f"\nworst overall: {worst:.3e}   "
          f"{'PASS' if worst < RTOL_PASS else 'FAIL'}")
    return worst


# Check 2: PyTorch autograd
def check_against_pytorch(seed=0):
    import torch
    import torch.nn as nn

    X, y, net = make_toy(seed=seed)
    analytic = net.backward(net.forward(X), y)

    d, h = net.W1.shape

    torch_net = nn.Sequential(
        nn.Linear(d, h),
        nn.Tanh(),
        nn.Linear(h, 1),
    ).double()        


    with torch.no_grad():
        torch_net[0].weight.copy_(torch.tensor(net.W1.T))
        torch_net[0].bias.copy_(torch.tensor(net.b1))
        torch_net[2].weight.copy_(torch.tensor(net.W2.T))
        torch_net[2].bias.copy_(torch.tensor(net.b2))

    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)

    loss = nn.MSELoss()(torch_net(Xt), yt)   
    loss.backward()                          

    pairs = {
        "W1": (analytic["W1"], torch_net[0].weight.grad.numpy().T),
        "b1": (analytic["b1"], torch_net[0].bias.grad.numpy()),
        "W2": (analytic["W2"], torch_net[2].weight.grad.numpy().T),
        "b2": (analytic["b2"], torch_net[2].bias.grad.numpy()),
    }

    print("\n--- PyTorch autograd check ---")
    worst = 0.0
    for name, (mine, theirs) in pairs.items():
        assert mine.shape == theirs.shape, f"{name}: {mine.shape} vs {theirs.shape}"
        err = float(np.max(np.abs(mine - theirs)) /
                    max(np.max(np.abs(mine)) + np.max(np.abs(theirs)), 1e-8))
        worst = max(worst, err)
        print(f"  {name:>3}  shape {str(mine.shape):>8}   relative error {err:.3e}")

    print(f"\nworst overall: {worst:.3e}   "
          f"{'PASS' if worst < RTOL_PASS else 'FAIL'}")
    return worst


if __name__ == "__main__":
    e1 = check_finite_differences()
    e2 = check_against_pytorch()

    print("\n" + "=" * 55)
    if max(e1, e2) < RTOL_PASS:
        print("ALL CHECKS PASSED: hand-derived gradients are correct.")
    else:
        print("CHECKS FAILED: the backward pass has a bug.")
    print("=" * 55)
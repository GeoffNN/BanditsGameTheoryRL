import numpy as np


def soda_strategy_discount(n_energy, n_nosugar):
    if n_energy <= n_nosugar:
        discount = -0.2;
    else:
        discount = 0.2;

    return discount


def soda_strategy_nodiscount(n_energy, n_nosugar):
    discount = 0.0;
    return discount


def soda_strategy_param(n_energy, n_nosugar, T, V):
    assert len(T) == len(V)
    diff = abs(n_energy - n_nosugar)
    sg = np.sign(n_energy - n_nosugar)
    if diff < T[0]:
        return 0
    if T[0] < diff <= T[1]:
        return sg * V[0]
    if T[1] < diff <= T[2]:
        return sg * V[1]
    if T[2] < diff:
        return sg * V[2]
    return 0

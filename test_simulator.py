from simulator import Simulator
from soda_strategy import soda_strategy_discount, soda_strategy_param, soda_strategy_nodiscount
import numpy as np

# Test simulator
s = Simulator()

R = 500  # nb of restocks to estimate the expectation
tot_rewards = np.zeros(R)
tot_sodas = np.zeros(R)

T = [2, 5, 10]
V = [.1, .3, .5]
# policy = lambda n_e, n_n: soda_strategy_param(n_e, n_n, T, V)
# policy = soda_strategy_nodiscount
policy = soda_strategy_discount

for r in range(R):
    s.reset()  # refill
    rew, n_energy, n_nosugar = s.simulate(0)
    reward = rew
    t = 0
    while n_energy > 0 and n_nosugar > 0:
        # no re-fill is needed
        discount = policy(n_energy, n_nosugar)
        rew, n_energy, n_nosugar = s.simulate(discount)
        reward = reward + rew
        t += 1
    tot_rewards[r] = reward  # total reward obtained
    tot_sodas[r] = t  # number of sodas sold

print(tot_sodas.mean())
print(tot_rewards.mean())
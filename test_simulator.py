from bandits import BanditVendingMachine
import matplotlib.pyplot as plt
from simulator import Simulator
from soda_strategy import soda_strategy_discount, soda_strategy_param, soda_strategy_nodiscount
import numpy as np
#
# # QUESTION 1
#
# # Test simulator
# s = Simulator()
#
# R = 500  # nb of restocks to estimate the expectation
# tot_rewards = np.zeros(R)
# tot_sodas = np.zeros(R)
# T = [2, 5, 10]
# V = [.1, .3, .5]
# # policy = lambda n_e, n_n: soda_strategy_param(n_e, n_n, T, V)
# # policy = soda_strategy_nodiscount
# policy = soda_strategy_discount
#
# for r in range(R):
#     s.reset()  # refill
#     rew, n_energy, n_nosugar = s.simulate(0)
#     reward = rew
#     t = 0
#     while n_energy > 0 and n_nosugar > 0:
#         # no re-fill is needed
#         discount = policy(n_energy, n_nosugar)
#         rew, n_energy, n_nosugar = s.simulate(discount)
#         reward = reward + rew
#         t += 1
#     tot_rewards[r] = reward  # total reward obtained
#     tot_sodas[r] = t  # number of sodas sold
#
# print(tot_sodas.mean())
# print(tot_rewards.mean())
#
# # QUESTION 2
#
# # Test simulator
# s = Simulator()
#
# R = 500  # nb of restocks to estimate the expectation
# tot_rewards = np.zeros(R)
# tot_sodas = np.zeros(R)

policies_params = [([1, 2, 3], [.1, .2, .3]),
                   ([1, 3, 6], [.1, .2, .3]),
                   ([2, 4, 6], [.1, .2, .3]),
                   ([3, 6, 9], [.2, .3, .4]),
                   ([5, 9, 15], [.05, .1, .4])]

banditVM = BanditVendingMachine(policies_params)

rws_ucb, draws_ucb = banditVM.UCB1(1000)
rws_TS, draws_TS = banditVM.TS(1000)
print("Done")

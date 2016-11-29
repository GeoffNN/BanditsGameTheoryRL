from simulator import Simulator
from soda_strategy import soda_strategy_discount, soda_strategy_nodiscount

s = Simulator()
R = 500

print(s.n_energy, s.n_nosugar)

rew, n_energy, n_nosugar = s.simulate(0)
s.reset()

V = [0.1, 0.2, 0.5]
T = [2, 5, 10]

policy1 = soda_strategy_discount
policy2 = soda_strategy_nodiscount
# policy3=@(n1,n2) soda_strategy_param(n1,n2,T,V)


# Choose a bandit problem

TableV = []
TableT = []

import arms
import numpy as np
from scipy.stats import bernoulli, beta

from arms import ArmVendingMachine
from simulator import Simulator


class Bandit:
    """Class implementing a generic Bandit. Can be constructed from a list of arms or
    as a sum of Bandits. All the following classes inherit from the generic Bandit."""

    def __init__(self, arms=None):
        if arms is not None:
            self.arms = arms
            self.n_arms = len(arms)
            self.means = [arm.mean for arm in arms]

    def __add__(self, other):
        return Bandit(arms=self.arms + other.arms)

    def sample(self, k):
        return self.arms[k].sample()

    def UCB1(self, time_horizon=1000):
        rews = np.zeros(time_horizon)
        draws = np.zeros(time_horizon)
        draws_per_bandit = np.zeros(self.n_arms)
        means = np.zeros(self.n_arms)
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = np.argmax((means + np.sqrt(np.log(t) / (2 * draws_per_bandit))))

            rews[t] = self.sample(arm_to_play)
            draws_per_bandit[arm_to_play] += 1
            means[arm_to_play] = ((draws_per_bandit[arm_to_play] - 1) * means[arm_to_play] +
                                  rews[t]) / draws_per_bandit[arm_to_play]
            draws[t] = arm_to_play

        return rews, draws

    def generalTS(self, time_horizon=1000):
        # print("Init TS")
        rews = np.zeros(time_horizon)
        S = np.zeros(self.n_arms)
        draws = np.zeros(time_horizon)
        N = np.zeros(self.n_arms)
        # print("Begin init loop")
        for t in range(time_horizon):
            arm_to_play = np.array([beta.rvs(S[a] + 1, N[a] - S[a] + 1)
                                    for a in range(self.n_arms)]).argmax()

            rews[t] = self.sample(arm_to_play)
            S[arm_to_play] += bernoulli.rvs(rews[t])
            N[arm_to_play] += 1
            draws[t] = arm_to_play
        return rews, draws

    def TS(self, time_horizon=100):
        # print("Init TS")
        rews = np.zeros(time_horizon)
        S = np.zeros(self.n_arms)
        draws = np.zeros(time_horizon)
        N = np.zeros(self.n_arms)
        # print("Begin loop")
        for t in range(time_horizon):
            arm_to_play = np.array([beta.rvs(S[a] + 1, N[a] - S[a] + 1)
                                    for a in range(self.n_arms)]).argmax()
            rews[t] = self.sample(arm_to_play)
            S[arm_to_play] += rews[t]
            N[arm_to_play] += 1
            draws[t] = arm_to_play

        return rews, draws

    def naive(self, time_horizon=1000):
        rews = np.zeros(time_horizon)
        draws = np.zeros(time_horizon)
        draws_per_bandit = np.zeros(self.n_arms)
        means = np.zeros(self.n_arms)
        for t in range(time_horizon):
            if t < self.n_arms:
                arm_to_play = t
            else:
                arm_to_play = means.argmax()
            rews[t] = self.sample(arm_to_play)
            draws_per_bandit[arm_to_play] += 1
            means[arm_to_play] = ((draws_per_bandit[arm_to_play] - 1) * means[arm_to_play] + rews[
                t]) / draws_per_bandit[arm_to_play]
            draws[t] = arm_to_play

        return rews, draws

    def complexity(self):
        p_star = max(self.means)
        return np.sum([(p_star - p) / self.kl(p, p_star) if p != p_star else 0 for p in self.means])

    def LRlowerbound(self, t):
        return self.complexity() * np.log(t)

    @staticmethod
    def kl(x, y):
        return x * np.log(y) + (1 - x) * np.log((1 - x) / (1 - y))


class BanditBernoulli(Bandit):
    def __init__(self, means=None, n_arms=None):
        Bandit.__init__(self)
        if means is None:
            if n_arms is None:
                self.n_arms = 10
            else:
                self.n_arms = n_arms
            self.means = [1 / (k + 1) for k in range(1, self.n_arms + 1)]
        else:
            self.means = means
            self.n_arms = len(self.means)
        self.arms = [arms.ArmBernoulli(p) for p in self.means]

    def __repr__(self):
        return self.means.__repr__()


class BanditBeta(Bandit):
    def __init__(self, parameter_list=None):
        if parameter_list is None:
            self.parameter_list = [(1, 1)]
        else:
            self.parameter_list = parameter_list
        self.arms = [arms.ArmFBeta(a, b) for a, b in parameter_list]
        self.n_arms = len(self.arms)

    def __repr__(self):
        return self.parameter_list__repr__()


class BanditExp(Bandit):
    def __init__(self, lambdas=None, n_arms=None):
        Bandit.__init__(self)
        if lambdas is None:
            if n_arms is None:
                self.n_arms = 2
            else:
                self.n_arms = n_arms
            self.lambdas = np.linspace(1, self.n_arms, self.n_arms)
        else:
            self.lambdas = lambdas
            self.n_arms = len(lambdas)
        self.arms = [arms.ArmExp(lambd) for lambd in self.lambdas]

    def __repr__(self):
        return self.lambdas.__repr__()


class BanditFinite(Bandit):
    def __init__(self, parameter_list=None):
        if parameter_list is None:
            self.parameter_list = [(0, 1)]
        else:
            self.parameter_list = parameter_list
        self.arms = [arms.ArmFinite(x, p) for x, p in parameter_list]
        self.n_arms = len(self.arms)

    def __repr__(self):
        return self.parameter_list.__repr__()

class BanditVendingMachine(Bandit):

    def __init__(self, param_list):
        self.arms = [ArmVendingMachine(params) for params in param_list]
        self.n_arms = len(self.arms)


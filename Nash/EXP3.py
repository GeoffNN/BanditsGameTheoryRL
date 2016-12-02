import numpy as np
from scipy.stats import bernoulli, rv_discrete


class EXP3:
    def __init__(self, eta=.1, beta=.1, nbActions=2):
        self.nbActions = nbActions
        self.eta = eta
        self.beta = beta
        self.p = np.array([1 / nbActions for _ in range(nbActions)])
        self.w = np.ones(self.nbActions)

    def init(self):
        self.w = np.ones(self.nbActions)

    def play(self):
        rand = bernoulli.rvs(self.beta)
        if rand:
            return np.random.randint(0, self.nbActions)
        return simu(self.p)

    def getReward(self, arm, reward):
        R = reward / (self.beta / self.nbActions + (1 - self.beta) * self.p[arm])
        self.w[arm] *= np.exp(self.eta * R)
        self.p[arm] = self.w[arm] / self.w.sum()
        # Seems there's underflow: the sum of p doesn't equal 1. Renormalize:
        self.p /= self.p.sum()


def simu(p):
    # random drawn under p: P(X=i)=p_i
    q = np.cumsum(p)
    u = np.random.rand()

    i = 0
    while u > q[i]:
        i += 1

    return i

import numpy as np


class EXP3():
    def __init__(self, eta, beta, nbActions):
        self.nbActions = nbActions
        self.eta = eta
        self.beta = beta
        self.w = np.ones(self.nbActions)

    def init(self):
        self.w = np.ones(self.nbActions)

    def play(self):
        return action

    def getReward(self, arm, reward):
        pass


def simu(p):
    # random drawn under p: P(X=i)=p_i
    q = np.cumsum(p);
    u = np.random.rand();

    i = 1;
    while u > q[i]:
        i = i + 1;

    return i
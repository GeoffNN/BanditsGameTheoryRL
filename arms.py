# ########################################################################### #
#     MVA -- Reinforcement Learning -- TP2
# ########################################################################### #
#
# Code base for the TP2 of the MVA lecture Reinforcement Learning, by
# Alessandro Lazaric. This is a Python port of the MATLAB base provided by the
# TP advisor Émilie Kaufmann:
# http://chercheurs.lille.inria.fr/ekaufman/teaching.html
#
# Do not hesitate to report any suggestion or bugfix to
# Élie Michel <elie.michel@ens.fr>
# 
# 
# Usage:
# from codeTP2 import *
# a = ArmBernoulli(0.3)
# a.sample()
#
# ########################################################################### #
#
# This piece of software is released under the MIT License:
#
# Copyright (c) 2016 Élie Michel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ########################################################################### #

import numpy as np
from numpy import exp, log
from numpy.random import random, beta

from simulator import Simulator
from soda_strategy import soda_strategy_param


class ArmBernoulli():
    """Bernoulli arm"""

    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        self.p = p
        self.mean = p
        self.var = p * (1 - p)

    def sample(self):
        reward = int(random() < self.p)
        return reward


class ArmBeta():
    """arm having a Beta distribution"""

    def __init__(self, a, b):
        """
        a: first beta parameter
        b: second beta parameter
        """
        self.a = a
        self.b = b
        self.mean = a / (a + b)
        self.var = (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self):
        reward = beta(self.a, self.b)
        return reward


class ArmExp():
    """arm with trucated exponential distribution"""

    def __init__(self, lambd):
        """
        lambd: parameter of the exponential distribution
        """
        self.lambd = lambd
        self.mean = (1 / lambd) * (1 - exp(-lambd))
        self.var = 1  # compute it yourself!

    def sample(self):
        reward = min(-1 / self.lambd * log(random()), 1)
        return reward

    def simu(p):
        """
        draw a sample of a finite-supported distribution that takes value
        k with porbability p(k)
        p: a vector of probabilities
        """
        q = p.cumsum()
        u = random()
        i = 0
        while u > q[i]:
            i += 1
            if i >= len(q):
                raise ValueError("p does not sum to 1")
        return i


class ArmFinite():
    """arm with finite support"""

    def __init__(self, X, P):
        """
        X: support of the distribution
        P: associated probabilities
        """
        self.X = np.array(X)
        self.P = np.array(P)
        self.mean = (self.X * self.P).sum()
        self.var = (self.X ** 2 * self.P).sum() - self.mean ** 2

    def sample(self):
        i = simu(self.P)
        reward = self.X[i]
        return reward


class ArmVendingMachine():
    def __init__(self, params=([1, 2, 3], [.1, .2, .3]), simulator=Simulator()):
        self.params = params
        self.simulator = simulator

    def sample(self):
        T, V = self.params
        policy = lambda n_e, n_n: soda_strategy_param(n_e, n_n, T, V)
        self.simulator.reset()  # refill
        rew, n_energy, n_nosugar = self.simulator.simulate(0)
        reward = rew
        t = 1
        while n_energy > 0 and n_nosugar > 0:
            # no re-fill is needed
            # Choose policy here
            discount = policy(n_energy, n_nosugar)
            rew, n_energy, n_nosugar = self.simulator.simulate(discount)
            reward = reward + rew
            t += 1
        return reward/100

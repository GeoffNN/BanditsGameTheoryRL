import bandits
import arms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


def q1(time_horizon=1000):
    rewards_ucb, draws_ucb = bandit.UCB1(time_horizon)
    plt.figure()
    draws_ucb.plot(style='.', title='UCB1')

    rewards_ts, draws_ts = bandit.TS(time_horizon)
    plt.figure()
    draws_ts.plot(style='.', title='Thompson Sampling')


def q2(bandit=bandits.BanditBernoulli(), n_samples=100, time_horizon=25000, figtitle="regret_curves"):
    p_star = max(bandit.means)
    regret_ucb = pd.Series(np.zeros(time_horizon))
    regret_ts = pd.Series(np.zeros(time_horizon))
    regret_general_ts = pd.Series(np.zeros(time_horizon))
    # regret_naive = pd.Series(np.zeros(time_horizon))

    for k in range(n_samples):
        # print(k)
        # print("Computing UCB")
        t0 = time.time()
        rewards_ucb, _ = bandit.UCB1(time_horizon)
        # print("Computing TS")
        rewards_ts, _ = bandit.TS(time_horizon)
        # print("Computing general TS")
        rewards_general_ts, _ = bandit.generalTS(time_horizon)
        # rewards_naive, _ = bandit.naive(time_horizon)
        # print("Computing regret")
        regret_ucb -= rewards_ucb.cumsum()
        regret_ts -= rewards_ts.cumsum()
        regret_general_ts -= rewards_general_ts.cumsum()
        # regret_naive -= rewards_naive.cumsum()

    regret_ucb /= n_samples
    regret_ts /= n_samples
    regret_general_ts /= n_samples
    # regret_naive /= n_samples

    opt = pd.Series(np.linspace(1, time_horizon, time_horizon)) * p_star
    regret_ucb += opt
    regret_ts += opt
    regret_general_ts += opt
    # regret_naive += opt

    regret_oracle = pd.Series([bandit.complexity() * np.log(t) for t in range(time_horizon)])

    fig = plt.figure()
    regret_ucb.plot(label='UCB regret')
    regret_ts.plot(label='Bernoulli Thompson Sampling regret')
    regret_general_ts.plot(label='General Thompson Sampling regret')
    # regret_naive.plot(label='Naive algorithm regret')
    regret_oracle.plot(label='Oracle regret')

    plt.legend(loc=4)
    plt.title('Regret curves')
    fig.savefig(figtitle + ".png")


start = time.time()
bandit = bandits.Bandit([arms.ArmExp(.4), arms.ArmExp(.5), arms.ArmBernoulli(.8), arms.ArmBernoulli(.9)])
q2(bandit)
print(time.time() - start)

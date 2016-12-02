from Nash.EXP3 import EXP3
import numpy as np


def EXP3vEXP3(n=100, eta=.05, beta=.1, G=None):
    playerA = EXP3(eta, beta, len(G))
    playerB = EXP3(eta, beta, len(G))
    actionsA = np.zeros(n)
    actionsB = np.zeros(n)
    rewards = np.zeros(n)
    for k in range(n):
        armA = playerA.play()
        armB = playerB.play()
        actionsA[k] = armA
        actionsB[k] = armB
        reward = G[armA, armB]
        rewards[k] = reward
        playerA.getReward(armA, reward)
        playerB.getReward(armB, -reward)

    return actionsA, actionsB, rewards
    # 1/4 for A, 1/2 for B

import numpy as np
import matplotlib.pyplot as plt

from Nash.adversarialEXP3 import EXP3vEXP3

G = np.array([[2, -1], [0, 1]])
actionsA, actionsB, rewards = EXP3vEXP3(100000, eta=.01, beta=.1, G=G)

print((actionsA == 0).mean())
print((actionsB == 0).mean())
# fig = plt.figure()
# plt.plot(rewards, '.')
# fig = plt.figure()
# plt.plot(actionsA)
# plt.plot(actionsB)

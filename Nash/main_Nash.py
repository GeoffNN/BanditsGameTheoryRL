import numpy as np
import matplotlib.pyplot as plt

from Nash.adversarialEXP3 import EXP3vEXP3


def cummean(A, axis=-1):
    """ Cumulative averaging

    Parameters
    ----------
    A    : input ndarray
    axis : axis along which operation is to be performed

    Output
    ------
    Output : Cumulative averages along the specified axis of input ndarray
    Found on StackOverflow: http://stackoverflow.com/questions/36409596/vectorize-numpy-mean-across-the-slices-of-an-array
    """

    return np.true_divide(A.cumsum(axis), np.arange(1, A.shape[axis] + 1))


G = np.array([[2, -1], [0, 1]])
actionsA, actionsB, rewards = EXP3vEXP3(100000, eta=.01, beta=.1, G=G)

print((actionsA == 0).mean())
print((actionsB == 0).mean())

plt.plot(cummean(actionsA == 0), label='p_a')
plt.plot(cummean(actionsB == 0), label='p_b')
plt.legend()

plt.plot(rewards)

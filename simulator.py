import numpy as np
from scipy.stats import bernoulli

class Simulator:
    """Simulates an arm with finite support"""

    def __init__(self, state=0, p_exam_no_exam=0.7, std_price=1, n_energy=50, n_nosugar=50):
        self.state = state
        self.p_exam_no_exam = p_exam_no_exam
        self.std_price = std_price
        self.n_energy = n_energy
        self.n_nosugar = n_nosugar

    def reset(self):
        self.n_energy = 50
        self.n_nosugar = 50
        self.state = 0

    def simulate(self, discount):
        discount_fraction = discount / self.std_price
        if not self.state:
            pref_energy = 0.6
            pref_nosugar = 1 - pref_energy

            if discount_fraction > 0:
                # The energy drink is discounted
                pref_energy = pref_energy * np.exp(2 * discount_fraction * np.log(1 / pref_energy))
                pref_nosugar = 1 - pref_energy

            elif discount_fraction < 0:  # The sugar free is discounted
                pref_nosugar = pref_nosugar * np.exp(-2 * discount_fraction * np.log(1 / pref_nosugar))
                pref_energy = 1 - pref_nosugar
        elif self.state == 1:
            # Exam situation
            pref_energy = 0.8
            pref_nosugar = 1 - pref_energy

            # Apply changed depending on discount
            if discount_fraction > 0:
                # Energy drink is discounted
                if 4 * pref_energy > 1:
                    pref_energy += (1 - pref_energy) * pref_energy ** 4
                else:
                    pref_energy += 3 * pref_energy * pref_energy ** 4
                pref_nosugar = 1 - pref_energy
            elif discount_fraction < 0:
                if (4 * pref_nosugar > 1.0):
                    pref_nosugar += (1 - pref_nosugar) * pref_nosugar ** 4
                else:
                    pref_nosugar += 3 * pref_nosugar * pref_nosugar ** 4
                end
                pref_energy = 1 - pref_nosugar
        # Random user preference
        rand = bernoulli.rvs(pref_energy)
        if rand:
            # User with preference for non sugar drink
            reward = self.std_price - max(discount, 0)
            self.n_energy -= 1
        else:
            # User with preference for energy drink
            reward = self.std_price + min(discount, 0)
            self.n_nosugar -= 1

            # Evolution of the state of the environment
        if self.state ==0 and bernoulli.rvs(self.p_exam_no_exam):
            self.state = 1
        elif self.state == 1 and bernoulli.rvs(self.p_exam_no_exam):
            self.state = 0

        return reward, self.n_energy, self.n_nosugar

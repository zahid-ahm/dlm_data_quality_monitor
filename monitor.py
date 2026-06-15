import numpy as np
from scipy.stats import norm

class Monitor:
    def __init__(self, k=3, tau=100):
        self.k = k      # size of shift the alternative looks for, in sd
        self.tau = tau  # evidence needed before flagging
        self.C = 1.0

    def check(self, y, f, S):
        e = abs(y - f) / np.sqrt(S)
        p = 2 * norm.sf(e)
        # Bayes factor of a k-sd shift against the model, accumulated
        e = min(e, 40) # avoid exp overflow on extreme errors
        B = np.exp(self.k * e - 0.5 * self.k ** 2)
        self.C = B * max(self.C, 1)
        alert = self.C > self.tau
        if alert:
            self.C = 1.0
        return alert, p, B

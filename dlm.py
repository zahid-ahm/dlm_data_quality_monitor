import numpy as np
from scipy.linalg import block_diag

def rot(j):
    w = 2 * np.pi * j / 7
    return np.array([[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]])

# linear trend + full weekly seasonal in Fourier form
F = block_diag(np.array([[1, 1], [0, 1]]), rot(1), rot(2), rot(3))
H = np.array([1.0, 0, 1, 0, 1, 0, 1, 0])
I = np.eye(8)

class DLM:
    def __init__(self, m0, V, discount=0.97):
        self.m = np.zeros(8)
        self.m[0] = m0
        self.P = I * V * 10
        self.V = V
        self.d = discount
        self.extra = 0.0

    # discounting plays the role of U
    def predict(self):
        m = F @ self.m
        P = F @ self.P @ F.T / self.d + self.extra * I
        f = H @ m
        S = H @ P @ H + self.V
        return m, P, f, S

    def update(self, y):
        m, P, f, S = self.predict()
        self.extra = 0.0
        if y is None:
            self.m, self.P = m, P  # missing day, carry the prediction
            return f, S
        z = y - f
        K = P @ H / S
        self.m = m + K * z
        self.P = P - np.outer(K, K) * S
        return f, S

    def intervene(self, var):
        self.extra = var


import numpy as np

class LinearRegression:
    def __init__(self, w=None, b=None):
        self.w = w
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b
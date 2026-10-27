import numpy as np
from math import e
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z = 1 / (1 + np.exp(-z))
        return np.round(z, 5)

    def relu(self, z: 
NDArray[np.float64]) -> NDArray[np.float64]:
        np.maximum(z, 0, out=z)
        np.round(z, 5, out=z)
        return z

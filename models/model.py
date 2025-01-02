from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X) -> np.array:
        pass

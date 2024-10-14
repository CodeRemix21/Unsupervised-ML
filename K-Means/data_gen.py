from sklearn import datasets
import numpy as np
import random

class Blobs:
    def gen_dataset(self, n_samples: int = 1, n_features: int = 1, random_state : int = 0):
        self.X, self.y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, random_state=random_state)
        return self.X

class RandomDataset:
    def gen_dataset(self, n_samples: int = 1, n_features: int = 1, mean: int = 10, std: int = 5):
        self.X = np.array([random.gauss(mean, std) for _ in range(n_samples*n_features)]).reshape(n_samples, n_features)
        return self.X






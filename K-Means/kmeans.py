import random
import numpy as np

class Cluster:
    def __init__(self):
        self.Centre = tuple()
        self.Points = np.array([])
        self.Distance = np.array([])

    def initialize_centroid(self, data: np.ndarray):
        self.Centre = tuple(data[random.randint(0, data.shape[0] - 1)])

    def update_centroid(self):
        x_mean = self.Points[:,0].mean()
        y_mean = self.Points[:,1].mean()
        new_centre = [x_mean, y_mean]
        self.Centre = tuple(new_centre)

    def update_points(self, points: np.ndarray):
        self.Points = points

    def calculate_distance_to_points(self, points: np.ndarray):
        x_axis = np.square(points[:,0]-self.Centre[0])
        y_axis = np.square(points[:,1]-self.Centre[1])
        self.Distance = np.sqrt(x_axis + y_axis)


class KMeans:
    def __init__(self, data: np.ndarray, n_clusters: int = 2):
        self.Distances = np.zeros((data.shape[0], n_clusters))
        self.Data = data
        self.Clusters = [Cluster() for _ in range(n_clusters)]

    def initialize_clusters(self):
        while len(set(np.argmin(self.Distances, axis=1))) != len(self.Clusters):
            for index, cluster in enumerate(self.Clusters):
                cluster.initialize_centroid(data=self.Data)
                cluster.calculate_distance_to_points(self.Data)
                self.Distances[:, index] = cluster.Distance
        return True

    def fit(self, n_inter: int = 1):
        if self.initialize_clusters():
            for _ in range(n_inter):
                for index, cluster in enumerate(self.Clusters):
                    cluster.calculate_distance_to_points(self.Data)
                    self.Distances[:, index] = cluster.Distance

                indices = np.argmin(self.Distances, axis=1)

                for index, cluster in enumerate(self.Clusters):
                    cluster.update_points(self.Data[indices == index])
                    cluster.update_centroid()


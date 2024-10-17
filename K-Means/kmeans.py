import random
import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    """
    Class to perform cluster behaviour

    Attributes:
        - centre:   tuple      - represents a centre of the cluster (X,Y) coordinates
        - points:   np.ndarray - represents set of points that belongs to the cluster
        - distance: np.ndarray - represents set of distances to each point in dataset

    Methods:
        - initialize_centroid          - set centroid as a random point from the dataset
        - update_centroid              - update centroid coordinates using mean value
        - update_points                - update set of points that belongs to the cluster
        - calculate_distance_to_points - calculate distance to each point from dataset using Euclidean Distance
    """

    def __init__(self):
        self.centre = tuple()
        self.points = np.array([])
        self.distance = np.array([])

    def initialize_centroid(self, data: np.ndarray):
        self.centre = tuple(data[random.randint(0, data.shape[0] - 1)])

    def update_centroid(self):
        x_mean = self.points[:, 0].mean()
        y_mean = self.points[:, 1].mean()
        new_centre = [x_mean, y_mean]
        self.centre = tuple(new_centre)

    def update_points(self, points: np.ndarray):
        self.points = points

    def calculate_distance_to_points(self, points: np.ndarray):
        x_axis = np.square(points[:,0] - self.centre[0])
        y_axis = np.square(points[:,1] - self.centre[1])
        self.distance = np.sqrt(x_axis + y_axis)


class KMeans:
    """
        Class to perform K-Means algorithm

        Attributes:
            - distances:  np.ndArray - represents matrix (size data_points x clusters) of distances from centroids to each point of dataset
            - labels:     np.ndArray - represents vector (size data_points) of colors for each point from dataset
            - indices:    np.ndArray - represents vector (size data_points) of indexes which indicates to which cluster points belong to
            - oldIndices: np.ndArray - copy of indices to detect end of algorithm
            - steps:      np.ndArray - represents matrix (size data_points x iterations) of indices to show progress in each iteration
            - data:       np.ndArray - input argument which represents dataset
            - clusters:   np.ndArray - set of clusters
            - cost:       np.ndArray - value of cost function in each iteration

        Methods:
            - check_clusters_points - check if clusters have at least one point
            - plt_colors            - calculate color values
            - initialize_clusters   - initialization process for centroids
            - show_progress         - plot progress of algorithm
            - fit                   - main loop of K-Means algorithm
        """

    def __init__(self, data: np.ndarray, n_clusters: int = 2):
        self.distances = np.zeros((data.shape[0], n_clusters))
        self.labels = np.zeros(data.shape[0])
        self.indices = np.zeros(data.shape[0])
        self.oldIndices = np.zeros(data.shape[0])
        self.steps = np.zeros((data.shape[0], 1), dtype=int)
        self.data = data
        self.clusters = [Cluster() for _ in range(n_clusters)]
        self.cost = np.array([])

    def check_clusters_points(self):
        return len(set(self.indices)) != len(self.clusters)

    def plt_colors(self):
        self.labels = self.indices / 255.

    def initialize_clusters(self):
        while self.check_clusters_points():
            for index, cluster in enumerate(self.clusters):
                cluster.initialize_centroid(data=self.data)
                cluster.calculate_distance_to_points(self.data)
                self.distances[:, index] = cluster.distance
            self.indices = np.argmin(self.distances, axis=1)
        self.steps[:,0] = self.indices
        return True

    def show_progress(self):
        fig, ax = plt.subplots(nrows=1, ncols=self.steps.shape[1])
        for i in range(self.steps.shape[1]):
            ax[i].scatter(self.data[:, 0], self.data[:, 1], c=(self.steps[:, i] / 255.))
            ax[i].set_title(f"Step {i}")
        fig.suptitle("K-Means algorithm progress")

    def fit(self):
        if self.initialize_clusters():
            while True:
                for index, cluster in enumerate(self.clusters):
                    cluster.update_points(self.data[self.indices == index])
                    cluster.update_centroid()
                    cluster.calculate_distance_to_points(self.data)
                    self.distances[:, index] = cluster.distance
                self.oldIndices = np.copy(self.indices)
                self.indices = np.argmin(self.distances, axis=1)
                self.steps = np.hstack((self.steps, np.array(self.indices).reshape((500, 1))))
                self.cost_function()
                if np.all(self.indices == self.oldIndices):
                    break
        self.plt_colors()

    def cost_function(self):
        j = 0
        for index, distance in enumerate(self.distances):
            j += distance[self.indices[index]]
        self.cost = np.append(self.cost, [j])

    def show_cost_function(self):
        plt.figure()
        plt.plot(range(self.cost.shape[0]), self.cost/np.max(self.cost))

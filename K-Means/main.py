import data_gen
import kmeans
import matplotlib.pyplot as plt

# generate dataset
blobs = data_gen.Blobs()
data = blobs.gen_dataset(500, 2, 7)
print(type(data), data.shape)

# Kmeans algorithm instance
k_means = kmeans.KMeans(data=data, n_clusters=3)

# plot data and randomly generated centroids
#plt.scatter(data[:,0], data[:,1])
#for cluster in k_means.clusters:
#    plt.scatter(cluster.centre[0], cluster.centre[1], c='k', marker='*', linewidths=4)
#plt.show()

k_means.fit()
plt.scatter(data[:,0], data[:,1], c=k_means.labels)
for cluster in k_means.clusters:
    plt.scatter(cluster.centre[0], cluster.centre[1], c='k', marker='*', linewidths=4)
k_means.show_progress()
k_means.show_cost_function()
plt.show()


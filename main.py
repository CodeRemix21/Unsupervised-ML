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
#for cluster in k_means.Clusters:
#    plt.scatter(cluster.Centre[0], cluster.Centre[1], c='k', marker='*', linewidths=4)
#plt.show()

k_means.fit(n_inter=100)
plt.scatter(k_means.Clusters[0].Points[:,0], k_means.Clusters[0].Points[:,1], c='b')
plt.scatter(k_means.Clusters[1].Points[:,0], k_means.Clusters[1].Points[:,1], c='g')
plt.scatter(k_means.Clusters[2].Points[:,0], k_means.Clusters[2].Points[:,1], c='y')
for cluster in k_means.Clusters:
    plt.scatter(cluster.Centre[0], cluster.Centre[1], c='k', marker='*', linewidths=4)
plt.show()


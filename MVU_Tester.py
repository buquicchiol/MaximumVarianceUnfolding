# mpl.use('Agg')
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from MVU import MaximumVarianceUnfolding

n = 550
dim = 2
k = 6

data, color = make_swiss_roll(n_samples=n)

# Calculate the nearest neighbors of each data point and build a graph
N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
N = np.array(N)

# Show the data

fig = plt.figure()
ax = Axes3D(fig)
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Original data")
for i in range(n):
    for j in range(n):
        if N[i, j] == 1.:
            ax.plot([data[i, 0], data[j, 0]],
                    [data[i, 1], data[j, 1]],
                    [data[i, 2], data[j, 2]],
                    'k',
                    linewidth=1,
                    linestyle='--')
ax.view_init(10, 90)
plt.savefig("Original_data.png", dpi=700)
plt.show()

start_time = time.clock()
mvu = MaximumVarianceUnfolding(equation="berkley")
embedded_data = mvu.fit_transform(data, dim, k, .15)
end_time = time.clock()

print("Total time: " + str(end_time - start_time))

plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=color)
plt.savefig("Embedded_data.png", dpi=700)
plt.show()

# TODO Show the convergence rate for different tolerances

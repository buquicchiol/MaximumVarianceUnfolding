# mpl.use('Agg')
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from MVU import MaximumVarianceUnfolding

n = 750
dim = 2
k = 15
noise = 0.0
dropout_rate = 0.65
max_iters = 1000
data_type = "s-curve"

filename = data_type + "_dim:" + str(dim) + "_dropout_rate:" + str(dropout_rate) + "_neighbors:" + str(
    k) + "_noise:" + str(noise) + "_n:" + str(n)

if data_type == "bent-rectangle":
    data = np.zeros((n, 3))
    for i in range(n):
        x = np.random.uniform(-1., 1.)
        z = np.random.uniform(-1., 1.)
        y = np.abs(x)

        data[i, :] = [x, y, z]
        _ = "b"

elif data_type == "s-curve":
    data, _ = make_s_curve(n, noise=noise, random_state=2)

elif data_type == "swiss-roll":
    data, _ = make_swiss_roll(n, noise=noise, random_state=2)

# Calculate the nearest neighbors of each data point and build a graph
N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
N = np.array(N)

for i in range(n):
    for j in range(n):
        if N[i, j] == 1 and np.random.random() < dropout_rate:
            N[i, j] = 0.

# Show the data

fig = plt.figure()
ax = Axes3D(fig)
ax = fig.gca(projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=_)
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
plt.tight_layout()
plt.savefig(("./Results/DropOutTest/" + "Original_data_" + filename + ".png"), dpi=700)
plt.show()

start_time = time.clock()
mvu = MaximumVarianceUnfolding(equation="berkley")
embedded_data = mvu.fit_transform(data, dim, k, dropout_rate)
end_time = time.clock()

print("Total time: " + str(end_time - start_time))

plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=_)
plt.tight_layout()
plt.savefig(("./Results/DropOutTest/" + "Embedded_data_" + filename + ".png"), dpi=700)
plt.show()

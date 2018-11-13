# mpl.use('Agg')
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skld
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

from MVU import MaximumVarianceUnfolding

n = 550
dim = 2
k = 10
noise = 0.0
dropout_rate = 0.0
max_iters = 2500
data_type = "bent-rectangle"
_3D = False

filename = "NOLANDMARK" + data_type + "_dim-" + str(dim) + "_dropout_rate-" + str(dropout_rate) + "_neighbors-" + str(
    k) + "_noise-" + str(noise) + "_n-" + str(n)

if data_type == "bent-rectangle":
    _3D = True
    data = np.zeros((n, 3))
    _ = []
    for i in range(int(n / 2)):
        x = np.random.uniform(-1., 1.)
        z = np.random.uniform(-1., 1.)
        y = np.abs(x)

        data[i, :] = [x, y, z]
        _.append("b")

    for j in range(int(n / 2)):
        x = np.random.uniform(-1., 1.)
        z = np.random.uniform(-1., 1.)
        y = -np.abs(x)

        data[(i + j), :] = [x, y, z]
        _.append("r")

elif data_type == "hemispheres":
    _3D = True
    data = np.zeros((n, 3))
    _ = []

    for i in range(n):
        x = np.random.normal(0, 1)
        y = np.random.normal(0, 1)
        z = np.random.normal(0, 1)

        norm = 1. / math.sqrt(x * x + y * y + z * z)

        if z > 0.:
            _.append(0)
            data[i, :] = [x * norm, y * norm, z * norm]
        else:
            _.append(0)
            data[i, :] = [x * norm, y * norm, -z * norm]
            # _.append(1)
            # data[i, :] = [x*norm/2, y*norm/2, -z*norm/2]

elif data_type == "s-curve":
    _3D = True
    data, _ = skld.make_s_curve(n, noise=noise, random_state=2)

elif data_type == "swiss-roll":
    _3D = True
    data, _ = skld.make_swiss_roll(n, noise=noise, random_state=2)

elif data_type == "circles":
    _3D = False
    data, _ = skld.make_circles(n, noise=noise, random_state=2)

# Calculate the nearest neighbors of each data point and build a graph
N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
N = np.array(N)
#
# # Sort the neighbor graph to find the points with the most connections
# num_connections = N.sum(axis=0).argsort()[::-1]
#
# # Separate the most popular points
# top_landmarks_idxs = num_connections[:40]
# top_landmarks = data[top_landmarks_idxs, :]
#
# # Compute the nearest neighbors for all of the landmarks so they are all connected
# L = NearestNeighbors(n_neighbors=3).fit(top_landmarks).kneighbors_graph(top_landmarks).todense()
# L = np.array(L)
#
# # The data without the landmarks
# new_data_idxs = [x for x in list(range(n)) if x not in top_landmarks_idxs]
# new_data = np.delete(data, top_landmarks_idxs, axis=0)
#
# # Construct a neighborhood graph where each point finds its closest landmark
# l = NearestNeighbors(n_neighbors=2).fit(top_landmarks).kneighbors_graph(new_data).todense()
# l = np.array(l)
#
# N = np.zeros((n, n))
#
# for i in range(40):
#     for j in range(40):
#         if L[i, j] == 1.:
#             N[top_landmarks_idxs[i], top_landmarks_idxs[j]] = 1.
#
# for i in range(510):
#     for j in range(40):
#         if l[i, j] == 1.:
#             N[new_data_idxs[i], top_landmarks_idxs[j]] = 1.

# Show the data
if _3D:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=_)
    ax.set_title("Original Data")
    for i in range(n):
        for j in range(n):
            # if (i < 40 and j < 40) and L[i, j] == 1.:
            #     ax.plot([data[i, 0], data[j, 0]],
            #             [data[i, 1], data[j, 1]],
            #             [data[i, 2], data[j, 2]],
            #             'k',
            #             linewidth=2,
            #             color='r',
            #             linestyle='--')
            if N[i, j] == 1.:
                ax.plot([data[i, 0], data[j, 0]],
                        [data[i, 1], data[j, 1]],
                        [data[i, 2], data[j, 2]],
                        'k',
                        linewidth=1,
                        linestyle='--')
    ax.view_init(65, 90)

else:
    plt.scatter(data[:, 0], data[:, 1], c=_)

plt.tight_layout()
plt.savefig(("./Results/" + "Original_data_" + filename + ".png"), dpi=700)
plt.show()

start_time = time.clock()
# imap = Isomap()
# embedded_data = imap.fit_transform(data)
mvu = MaximumVarianceUnfolding(equation="berkley", solver_iters=1000, warm_start=False)
embedded_data = mvu.fit_transform(data, dim, k, dropout_rate)
end_time = time.clock()

print("Total time: " + str(end_time - start_time))

plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=_)
plt.tight_layout()
plt.savefig(("./Results/" + "Embedded_data_" + filename + ".png"), dpi=700)
plt.show()

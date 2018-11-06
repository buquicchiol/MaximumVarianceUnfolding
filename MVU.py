import cvxpy as cp
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(threshold=np.nan)


class DisconnectError(Exception):
    """

    """

    def __init__(self, message):
        self.message = message


class MaximumVarianceUnfolding:

    def __init__(self, equation="berkley", solver=cp.SCS, solver_tol=1e-2, neg_tol=-1.0e-10, seed=None):
        """
        STUB
        :param equation:
        :param solver:
        :param solver_tol:
        :param neg_tol:
        :param seed:
        """
        self.equation = equation
        self.solver = solver
        self.solver_tol = solver_tol
        self.neg_tol = neg_tol
        self.seed = seed

    def fit(self, data, k, dropout_rate=.2):
        """
        The method to fit an MVU model to the data.

        :param data: The data to which the model will be fitted.
        :param k: The number of neighbors to fix
        :param dropout_rate: The number of neighbors to discount
        :return: Embedded Gramian: The Gramian matrix of the embedded data
        """
        # Number of data points in the set
        n = data.shape[0]

        # Set the seed
        np.random.seed(self.seed)

        # Calculate the nearest neighbors of each data point and build a graph
        N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
        N = np.array(N)

        # To check for disconnected regions in the neighbor graph
        lap = laplacian(N, normed=True)
        eigvals, _ = np.linalg.eig(lap)

        for e in eigvals:
            if e == 0.:
                raise DisconnectError("FATAL ERROR DISCONNECTED REGIONS IN NEIGHBORHOOD GRAPH")

        # Declare some CVXPy variables
        P = cp.Constant(data.dot(data.T))  # Gramian of the original data

        Q = cp.Variable((n, n), PSD=True)  # The projection of the Gramian
        Q.value = np.zeros((n, n))  # Initialized to zeros

        ONES = cp.Constant(np.ones((n, 1)))  # A shorter way to call a vector of 1's
        T = cp.Constant(n)  # A variable to keep the notation consistent with the Berkley lecture

        # Declare placeholders to get rid of annoying warnings
        objective = None
        constraints = []

        # Wikipedia Solution
        if self.equation == "wikipedia":
            objective = cp.Maximize(cp.trace(Q))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]

            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1:
                        constraints.append((P[i, i] + P[j, j] - P[i, j] - P[j, i]) -
                                           (Q[i, i] + Q[j, j] - Q[i, j] - Q[j, i]) == 0)

        # UC Berkley Solution
        if self.equation == "berkley":
            objective = cp.Maximize(cp.multiply((1 / T), cp.trace(Q)) -
                                    cp.multiply((1 / (T * T)), cp.trace(cp.matmul(cp.matmul(Q, ONES), ONES.T))))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]
            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1. and np.random.random() > dropout_rate:
                        constraints.append(Q[i, i] - 2 * Q[i, j] + Q[j, j] -
                                           (P[i, i] - 2 * P[i, j] + P[j, j]) == 0)

        # Solve the problem with the SCS Solver
        problem = cp.Problem(objective, constraints)
        # FIXME The solvertol syntax is unique to SCS
        problem.solve(solver=self.solver, eps=self.solver_tol)

        return Q.value

    def fit_transform(self, data, dim, k, dropout_rate=.2):
        """
        The method to fit and transform an MVU model to the data.

        :param data: The data to which the model will be fitted.
        :param dim: The new dimension of the dataset.
        :param k: The number of neighbors to fix
        :param dropout_rate: The number of neighbors to discount
        :return: embedded_data: The embedded form of the data.
        """

        # Number of data points in the set
        n = data.shape[0]

        # Set the seed
        np.random.seed(self.seed)

        # Calculate the nearest neighbors of each data point and build a graph
        N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
        N = np.array(N)

        # To check for disconnected regions in the neighbor graph
        lap = laplacian(N, normed=True)
        eigvals, _ = np.linalg.eig(lap)

        for e in eigvals:
            if e == 0.:
                raise DisconnectError("FATAL ERROR DISCONNECTED REGIONS IN NEIGHBORHOOD GRAPH")

        # Declare some CVXPy variables
        P = cp.Constant(data.dot(data.T))  # Gramian of the original data

        Q = cp.Variable((n, n), PSD=True)  # The projection of the Gramian
        Q.value = np.zeros((n, n))  # Initialized to zeros

        ONES = cp.Constant(np.ones((n, 1)))  # A shorter way to call a vector of 1's
        T = cp.Constant(n)  # A variable to keep the notation consistent with the Berkley lecture

        # Declare placeholders to get rid of annoying warnings
        objective = None
        constraints = []

        # Wikipedia Solution
        if self.equation == "wikipedia":
            objective = cp.Maximize(cp.trace(Q))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]

            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1:
                        constraints.append((P[i, i] + P[j, j] - P[i, j] - P[j, i]) -
                                           (Q[i, i] + Q[j, j] - Q[i, j] - Q[j, i]) == 0)

        # UC Berkley Solution
        if self.equation == "berkley":
            objective = cp.Maximize(cp.multiply((1 / T), cp.trace(Q)) -
                                    cp.multiply((1 / (T * T)), cp.trace(cp.matmul(cp.matmul(Q, ONES), ONES.T))))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]
            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1. and np.random.random() > dropout_rate:
                        constraints.append(Q[i, i] - 2 * Q[i, j] + Q[j, j] -
                                           (P[i, i] - 2 * P[i, j] + P[j, j]) == 0)

        # Solve the problem with the SCS Solver
        problem = cp.Problem(objective, constraints)
        # FIXME The solvertol syntax is unique to SCS
        problem.solve(solver=self.solver, eps=self.solver_tol)

        # Retrieve Q
        embedded_gramian = Q.value

        # Decompose gramian to recover the projection
        eigenvalues, eigenvectors = np.linalg.eig(embedded_gramian)

        eigenvalues[np.logical_and(self.neg_tol < eigenvalues, eigenvalues < 0.)] = 0.

        sorted_indices = eigenvalues.argsort()[::-1]
        top_eigenvalue_indices = sorted_indices[:dim]

        top_eigenvalues = eigenvalues[top_eigenvalue_indices]
        top_eigenvectors = eigenvectors[top_eigenvalue_indices, :]

        lbda = np.diag(top_eigenvalues ** 0.5)

        embedded_data = lbda.dot(top_eigenvectors).T

        return embedded_data

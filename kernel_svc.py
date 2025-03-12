import numpy as np
import itertools
from collections import Counter
from scipy.sparse import csr_matrix
from cvxopt import matrix, solvers

class KernelSVC:
    def __init__(self, C=1.0, epsilon=1e-5):
        """
        Optimized Kernel SVM using a Quadratic Programming solver.
        
        Parameters:
        - C: float, regularization parameter.
        - kernel: str, 'spectrum' or 'mismatch'.
        """
        self.C = C
        self.epsilon = epsilon

        self.alpha = None  # Dual coefficients
        self.support_vectors = None  # Stored feature vectors
        self.support_y = None
        self.b = None


    def fit(self, K, vectors, y):
        """Fits the SVM model using Quadratic Programming."""
        self.K = K
        self.vectors = vectors
        N = len(y)
        y = np.array(y).astype(float) * 2 - 1  # Convert to {-1, 1}

        # Construct QP problem for dual SVM
        P = matrix(np.outer(y, y) * self.K)
        q = matrix(-np.ones(N))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), self.C * np.ones(N))))
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(0.0)

        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        # Store support vectors
        sv_idx = alpha > self.epsilon
        self.sv_idx=sv_idx
        self.alpha = alpha[sv_idx]
        self.support_vectors = [self.vectors[i] for i in range(N) if sv_idx[i]]
        self.support_y = y[sv_idx]

        print(f"proportion of support vectors: {len(sv_idx)/N}")

        # Compute bias term
        self.b = np.mean([y[i] - np.sum(self.alpha * self.support_y * self.K[i, sv_idx])
                          for i in range(N) if sv_idx[i]])
        
        return self.support_vectors

    def separating_function(self, K_test):
        """Compute the separating function for new sequences."""
        return K_test[:,self.sv_idx] @ (self.alpha * self.support_y) + self.b

    def predict(self, K_test):
        """Predicts class labels (-1 or 1)."""
        return np.sign(self.separating_function(K_test))

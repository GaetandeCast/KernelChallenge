import numpy as np
from cvxopt import matrix, solvers

class KernelSVC:
    def __init__(self, C=1.0, epsilon=1e-5):
        """
        Initializes the Kernel Support Vector Classifier (SVC).
        
        Parameters:
            C (float): Regularization parameter.
            epsilon (float): Threshold to determine support vectors.
        """
        self.C = C
        self.epsilon = epsilon

        self.alpha = None  # Dual coefficients
        self.support_vectors = None  # Stored feature vectors
        self.support_y = None  # Labels of support vectors
        self.b = None  # Bias term

    def fit(self, K, vectors, y):
        """
        Fits the kernel SVM model using the provided kernel matrix.
        
        Parameters:
            K (ndarray): Precomputed kernel matrix.
            vectors (list or ndarray): Training data vectors.
            y (list or ndarray): Target labels (0 or 1).
        
        Returns:
            out (list): List of support vectors.
        """
        self.K = K
        self.vectors = vectors
        N = len(y)
        
        # Convert labels to {-1, 1} for SVM formulation
        y = np.array(y).astype(float) * 2 - 1  

        # Construct the Quadratic Programming (QP) problem
        P = matrix(np.outer(y, y) * self.K)  # Quadratic term
        q = matrix(-np.ones(N))  # Linear term
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))  # Inequality constraints
        h = matrix(np.hstack((np.zeros(N), self.C * np.ones(N))))  # Upper and lower bounds
        A = matrix(y.reshape(1, -1), tc='d')  # Equality constraint
        b = matrix(0.0)

        # Solve the QP problem using cvxopt
        solvers.options['show_progress'] = False  # Suppress solver output
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])  # Extract solution

        # Identify support vectors
        sv_idx = alpha > self.epsilon  # Find nonzero alphas
        self.sv_idx = sv_idx
        self.alpha = alpha[sv_idx]
        self.support_vectors = [self.vectors[i] for i in range(N) if sv_idx[i]]
        self.support_y = y[sv_idx]

        print(f"Proportion of support vectors: {len(self.support_vectors) / N:.2f}")

        # Compute bias term using support vectors
        self.b = np.mean([
            y[i] - np.sum(self.alpha * self.support_y * self.K[i, sv_idx])
            for i in range(N) if sv_idx[i]
        ])
        
        return self.support_vectors

    def separating_function(self, K_test):
        """
        Computes the decision function values for a given test kernel matrix.
        
        Parameters:
            K_test (ndarray): Test kernel matrix.
        
        Returns:
            out (ndarray): Decision function values.
        """
        return K_test[:, self.sv_idx] @ (self.alpha * self.support_y) + self.b

    def predict(self, K_test):
        """
        Predicts class labels for a given test kernel matrix.
        
        Parameters:
            K_test (ndarray): Test kernel matrix.
        
        Returns:
            out (ndarray): Predicted labels (0 or 1).
        """
        return np.sign(self.separating_function(K_test))

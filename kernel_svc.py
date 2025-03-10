import numpy as np
import itertools
from collections import Counter
from scipy.sparse import csr_matrix
from cvxopt import matrix, solvers

class KernelSVC:
    def __init__(self, C=1.0, kernel='spectrum', kmer_size=3, mismatch=0, epsilon=1e-9):
        """
        Optimized Kernel SVM using a Quadratic Programming solver.
        
        Parameters:
        - C: float, regularization parameter.
        - kernel: str, 'spectrum' or 'mismatch'.
        - kmer_size: int, length of k-mers to consider.
        - mismatch: int, max number of mismatches (only for 'mismatch' kernel).
        """
        if kernel not in ['spectrum', 'mismatch']:
            raise ValueError("Kernel must be 'spectrum' or 'mismatch'")
        self.C = C
        self.kernel = kernel
        self.kmer_size = kmer_size
        self.mismatch = mismatch
        self.epsilon = epsilon

        self.alpha = None  # Dual coefficients
        self.support_vectors = None  # Stored feature vectors
        self.support_y = None
        self.b = None

        self._neighbors_cache = {}

    def _neighbors(self, kmer, max_mismatch):
        """Returns a set of k-mers within max_mismatch of the given k-mer."""
        key = (kmer, max_mismatch)
        if key in self._neighbors_cache:
            return self._neighbors_cache[key]
        
        alphabet = "ACGT"
        neighbors = set()
        for candidate in itertools.product(alphabet, repeat=len(kmer)):
            candidate = ''.join(candidate)
            mismatches = sum(1 for a, b in zip(kmer, candidate) if a != b)
            if mismatches <= max_mismatch:
                neighbors.add(candidate)
        self._neighbors_cache[key] = neighbors
        return neighbors

    def _get_feature_vector(self, seq):
        """Efficiently computes feature vector as a sparse dictionary."""
        counter = Counter()
        n = len(seq)
        k = self.kmer_size
        if n < k:
            return counter  # Too short sequence
        
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            if self.kernel == 'spectrum':
                counter[kmer] += 1
            elif self.kernel == 'mismatch':
                neighbors = self._neighbors(kmer, self.mismatch)
                for nb in neighbors:
                    counter[nb] += 1
        return counter

    def _compute_kernel_matrix(self, fvecs1, fvecs2):
        """Optimized kernel matrix computation using sparse dot product."""
        n1, n2 = len(fvecs1), len(fvecs2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                common_kmers = set(fvecs1[i].keys()) & set(fvecs2[j].keys())
                K[i, j] = sum(fvecs1[i][k] * fvecs2[j][k] for k in common_kmers)
        return K

    def fit(self, sequences, y):
        """Fits the SVM model using Quadratic Programming."""
        N = len(y)
        y = np.array(y).astype(float) * 2 - 1  # Convert to {-1, 1}

        # Compute sparse feature vectors
        print("Computing feature vectors...")
        feature_vectors = [self._get_feature_vector(seq) for seq in sequences]

        # Compute kernel matrix
        print("Computing kernel matrix...")
        K = self._compute_kernel_matrix(feature_vectors, feature_vectors)

        # Construct QP problem for dual SVM
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(N))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), self.C * np.ones(N))))
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(0.0)

        # Solve QP problem
        solvers.options['show_progress'] = True
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        # Store support vectors
        sv_idx = alpha > self.epsilon
        self.alpha = alpha[sv_idx]
        self.support_vectors = [feature_vectors[i] for i in range(N) if sv_idx[i]]
        self.support_y = y[sv_idx]

        print(f"proportion of support vectors: {len(sv_idx)/N}")

        # Compute bias term
        self.b = np.mean([y[i] - np.sum(self.alpha * self.support_y * K[i, sv_idx])
                          for i in range(N) if sv_idx[i]])

    def separating_function(self, sequences):
        """Compute the separating function for new sequences."""
        test_fvecs = [self._get_feature_vector(seq) for seq in sequences]
        K = self._compute_kernel_matrix(test_fvecs, self.support_vectors)
        return K @ (self.alpha * self.support_y) + self.b

    def predict(self, X):
        """Predicts class labels (-1 or 1)."""
        return np.sign(self.separating_function(X))

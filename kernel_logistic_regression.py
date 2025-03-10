import numpy as np
import itertools
import pandas as pd


class KernelLogisticRegression:
    def __init__(self, kernel='spectrum', kmer_size=3, mismatch=0,
                 n_iter=100, tol=1e-6, reg=0.01):
        """
        Initializes the Kernel Logistic Regression model for DNA sequence classification.

        Parameters:
        - kernel: str, either 'spectrum' or 'mismatch'.
            'spectrum': uses exact k-mer counts.
            'mismatch': counts all k-mers within a given Hamming distance.
        - kmer_size: int, length of the k-mers to consider.
        - mismatch: int, maximum number of mismatches allowed (used only if kernel=='mismatch').
        - n_iter: int, maximum number of Newton-Raphson iterations.
        - tol: float, tolerance for convergence (based on the norm of the Newton update).
        - reg: float, L2 regularization strength.
        """
        if kernel not in ['spectrum', 'mismatch']:
            raise ValueError("kernel must be 'spectrum' or 'mismatch'")
        self.kernel = kernel
        self.kmer_size = kmer_size
        self.mismatch = mismatch  # used only for mismatch kernel
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        
        self.alpha = None       # Dual coefficients
        self.feature_vectors = None  # List of feature vectors for training sequences
        self.train_seqs = None       # Training sequences (raw)
        
        # Cache for neighbors computations (key: (kmer, mismatch) -> set of neighbors)
        self._neighbors_cache = {}

    def _neighbors(self, kmer, max_mismatch):
        """
        Returns a set of all k-mers that are within max_mismatch of the given kmer.
        This implementation uses a brute-force approach by generating all possible k-mers.
        """
        key = (kmer, max_mismatch)
        if key in self._neighbors_cache:
            return self._neighbors_cache[key]
        
        alphabet = "ACGT"
        neighbors = set()
        # Generate all possible k-mers of the same length
        for candidate in itertools.product(alphabet, repeat=len(kmer)):
            candidate = ''.join(candidate)
            # Compute Hamming distance
            mismatches = sum(1 for a, b in zip(kmer, candidate) if a != b)
            if mismatches <= max_mismatch:
                neighbors.add(candidate)
        self._neighbors_cache[key] = neighbors
        return neighbors

    def _get_feature_vector(self, seq):
        """
        Computes the feature vector for a given sequence as a dictionary mapping k-mer to count.
        For the spectrum kernel, counts exact k-mers.
        For the mismatch kernel, for each k-mer in the sequence, counts all its neighbors.
        """
        vec = {}
        n = len(seq)
        k = self.kmer_size
        if n < k:
            return vec  # sequence too short
        
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            if self.kernel == 'spectrum':
                # Count the exact k-mer
                vec[kmer] = vec.get(kmer, 0) + 1
            elif self.kernel == 'mismatch':
                # Count all neighbors within allowed mismatches
                neighbors = self._neighbors(kmer, self.mismatch)
                for nb in neighbors:
                    vec[nb] = vec.get(nb, 0) + 1
        return vec

    def _compute_kernel_matrix(self, fvecs1, fvecs2):
        """
        Computes the kernel matrix between two lists of feature vectors.
        The kernel value is the dot product between the two dictionaries.
        """
        n1 = len(fvecs1)
        n2 = len(fvecs2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # Dot product between dictionary fvecs1[i] and fvecs2[j]
                # Only iterate over keys in the first dict.
                dp = 0
                for key, val in fvecs1[i].items():
                    dp += val * fvecs2[j].get(key, 0)
                K[i, j] = dp
        return K

    def fit(self, sequences, y):
        """
        Fits the Kernel Logistic Regression model using the Newton-Raphson method.
        
        Parameters:
        - sequences: list or array of DNA sequences (strings) for training.
        - y: numpy array of shape (n_samples,), binary labels (0 or 1).
        """
        self.train_seqs = sequences[:]  # store raw sequences
        n_samples = len(sequences)
        y = np.array(y).flatten()
        
        # Precompute feature vectors for training sequences
        self.feature_vectors = [self._get_feature_vector(seq) for seq in sequences]
        
        # Compute the kernel matrix on training data
        K = self._compute_kernel_matrix(self.feature_vectors, self.feature_vectors)
        
        # Initialize dual coefficients
        self.alpha = np.zeros(n_samples)
        
        # Newton-Raphson iterations
        for iteration in range(self.n_iter):
            # Decision function f = K * alpha
            f = np.dot(K, self.alpha)
            # Sigmoid function to get probabilities
            p = 1 / (1 + np.exp(-f))
            # Gradient of the loss (with regularization)
            gradient = np.dot(K.T, (p - y)) + self.reg * self.alpha
            # Diagonal weight matrix W: p*(1-p)
            W = np.diag(p * (1 - p))
            # Hessian: K^T * W * K + reg * I
            Hessian = np.dot(K.T, np.dot(W, K)) + self.reg * np.eye(n_samples)
            
            # Solve for Newton update: Hessian * delta = gradient
            try:
                delta = np.linalg.solve(Hessian, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(Hessian, gradient, rcond=None)[0]
            
            # Update dual coefficients
            self.alpha -= delta
            
            # Check convergence
            if np.linalg.norm(delta) < self.tol:
                print(f"Converged in {iteration + 1} iterations.")
                break

    def predict_proba(self, sequences):
        """
        Predicts probabilities for the positive class for new sequences.
        
        Parameters:
        - sequences: list or array of DNA sequences (strings).
        
        Returns:
        - p: numpy array of predicted probabilities.
        """
        # Compute feature vectors for test sequences
        test_fvecs = [self._get_feature_vector(seq) for seq in sequences]
        # Compute the kernel matrix between test and training sequences
        K_test = self._compute_kernel_matrix(test_fvecs, self.feature_vectors)
        # Decision function
        f = np.dot(K_test, self.alpha)
        p = 1 / (1 + np.exp(-f))
        return p

    def predict(self, sequences):
        """
        Predicts binary labels (0 or 1) for new sequences.
        
        Parameters:
        - sequences: list or array of DNA sequences (strings).
        
        Returns:
        - predictions: numpy array of predicted binary labels.
        """
        proba = self.predict_proba(sequences)
        return (proba >= 0.5).astype(int)


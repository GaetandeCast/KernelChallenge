import numpy as np
import itertools
from collections import Counter
from scipy.sparse import csr_matrix
from cvxopt import matrix, solvers

class spectrum_kernel():
    def __init__(self, kmer_size=3, mismatch=None):
        self.kmer_size = kmer_size
        self.mismatch = mismatch

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

    def get_feature_vector(self, seq):
        """Efficiently computes feature vector as a sparse dictionary."""
        counter = Counter()
        n = len(seq)
        k = self.kmer_size
        if n < k:
            return counter  # Too short sequence
        
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            if self.mismatch == None:
                counter[kmer] += 1
            else:
                neighbors = self._neighbors(kmer, self.mismatch)
                for nb in neighbors:
                    counter[nb] += 1
        return counter

    def compute_kernel_matrix(self, fvecs1, fvecs2):
        """Optimized kernel matrix computation using sparse dot product."""
        n1, n2 = len(fvecs1), len(fvecs2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                common_kmers = set(fvecs1[i].keys()) & set(fvecs2[j].keys())
                K[i, j] = sum(fvecs1[i][k] * fvecs2[j][k] for k in common_kmers)
        return K
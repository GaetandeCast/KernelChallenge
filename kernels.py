import numpy as np
import itertools
from collections import Counter
import itertools
import numpy as np
from collections import Counter

class spectrum_kernel():
    def __init__(self, kmer_size=3, mismatch=None):
        """
        Initializes the spectrum kernel.
        
        Parameters:
        kmer_size (int): Size of the k-mers to extract.
        mismatch (int or None): Maximum number of mismatches allowed.
        """
        self.kmer_size = kmer_size
        self.mismatch = mismatch

        # Cache for storing previously computed neighbor k-mers
        self._neighbors_cache = {}

    def _neighbors(self, kmer, max_mismatch):
        """
        Generates all k-mers that are within the allowed mismatch limit.
        Uses caching to optimize repeated calculations.
        
        Parameters:
            kmer (str): The original k-mer.
            max_mismatch (int): Maximum number of mismatches allowed.
        
        Returns:
            out (set): A set of neighboring k-mers.
        """
        key = (kmer, max_mismatch)
        if key in self._neighbors_cache:
            return self._neighbors_cache[key]
        
        alphabet = "ACGT"  # DNA alphabet
        neighbors = set()
        
        # Generate all possible k-mers of the same length
        for candidate in itertools.product(alphabet, repeat=len(kmer)):
            candidate = ''.join(candidate)
            
            # Count mismatches between original k-mer and candidate
            mismatches = sum(1 for a, b in zip(kmer, candidate) if a != b)
            if mismatches <= max_mismatch:
                neighbors.add(candidate)
        
        # Store result in cache for faster future retrieval
        self._neighbors_cache[key] = neighbors
        return neighbors

    def compute_feature_vector(self, seq):
        """
        Computes the feature vector for a given sequence.
        
        Parameters:
            seq (str): DNA sequence.
        
        Returns:
            out (Counter): A dictionary-like object with k-mer counts.
        """
        counter = Counter()
        n = len(seq)
        k = self.kmer_size
        
        # If the sequence is shorter than the k-mer size, return empty counter
        if n < k:
            return counter  
        
        # Slide over the sequence and extract k-mers
        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            
            # If no mismatches are allowed, count exact k-mers
            if self.mismatch is None:
                counter[kmer] += 1
            else:
                # Otherwise, count all neighboring k-mers with allowed mismatches
                neighbors = self._neighbors(kmer, self.mismatch)
                for nb in neighbors:
                    counter[nb] += 1
        
        return counter

    def compute_kernel_matrix(self, fvecs1, fvecs2):
        """
        Computes the kernel matrix between two sets of feature vectors.
        
        Parameters:
            fvecs1 (list of Counter): Feature vectors for the first set of sequences.
            fvecs2 (list of Counter): Feature vectors for the second set of sequences.
        
        Returns:
            out (ndarray): Kernel matrix where K[i, j] represents similarity between fvecs1[i] and fvecs2[j].
        """
        n1, n2 = len(fvecs1), len(fvecs2)
        K = np.zeros((n1, n2))

        # Compute dot product between feature vectors
        for i in range(n1):
            for j in range(n2):
                common_kmers = set(fvecs1[i].keys()) & set(fvecs2[j].keys())
                K[i, j] = sum(fvecs1[i][k] * fvecs2[j][k] for k in common_kmers)
        
        return K

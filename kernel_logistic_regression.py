import numpy as np
import itertools
import pandas as pd


class KernelLogisticRegression:
    def __init__(self, n_iter=100, tol=1e-6, reg=0.01):
        """
        Initializes the Kernel Logistic Regression model for DNA sequence classification.

        Parameters:
        - n_iter: int, maximum number of Newton-Raphson iterations.
        - tol: float, tolerance for convergence (based on the norm of the Newton update).
        - reg: float, L2 regularization strength.
        """
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        
        self.alpha = None       # Dual coefficients
        self.feature_vectors = None  # List of feature vectors for training sequences
        self.train_seqs = None       # Training sequences (raw)
        

    def fit(self, K, y):
        """
        Fits the Kernel Logistic Regression model using the Newton-Raphson method.
        
        Parameters:
        - sequences: list or array of DNA sequences (strings) for training.
        - y: numpy array of shape (n_samples,), binary labels (0 or 1).
        """
        n_samples = len(y)
        y = np.array(y).flatten()
                
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

    def predict_proba(self, K_test):
        """
        Predicts probabilities for the positive class for new sequences.
        
        Parameters:
        
        Returns:
        - p: numpy array of predicted probabilities.
        """
        # Decision function
        f = np.dot(K_test, self.alpha)
        p = 1 / (1 + np.exp(-f))
        return p

    def predict(self, K_test):
        """
        Predicts binary labels (0 or 1) for new sequences.
        
        Parameters:
        - sequences: list or array of DNA sequences (strings).
        
        Returns:
        - predictions: numpy array of predicted binary labels.
        """
        proba = self.predict_proba(K_test)
        return (proba >= 0.5).astype(int)


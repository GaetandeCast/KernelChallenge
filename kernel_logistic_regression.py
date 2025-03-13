import numpy as np

class KernelLogisticRegression:
    def __init__(self, n_iter=100, tol=1e-6, reg=0.01):
        """
        Initializes the Kernel Logistic Regression model.
        
        Parameters:
            n_iter (int): Maximum number of iterations for optimization.
            tol (float): Tolerance for convergence.
            reg (float): Regularization strength.
        """
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        
        self.alpha = None 
        self.feature_vectors = None
        self.train_seqs = None  

    def fit(self, K, vectors, y):
        """
        Fits the Kernel Logistic Regression model using Newton-Raphson optimization.
        
        Parameters:
            K (ndarray): Precomputed kernel matrix.
            y (list or ndarray): Target labels (0 or 1).
        """
        n_samples = len(y)
        y = np.array(y).flatten()
                
        # Initialize dual coefficients
        self.alpha = np.zeros(n_samples)
        
        # Newton-Raphson iterations
        for iteration in range(self.n_iter):
            # Compute decision function f = K * alpha
            f = np.dot(K, self.alpha)
            # Compute probabilities using the sigmoid function
            p = 1 / (1 + np.exp(-f))
            # Compute gradient of the loss function with regularization
            gradient = np.dot(K.T, (p - y)) + self.reg * self.alpha
            # Compute diagonal weight matrix W with elements p * (1 - p)
            W = np.diag(p * (1 - p))
            # Compute Hessian matrix: K^T * W * K + reg * I
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
        Computes predicted probabilities for test data.
        
        Parameters:
            K_test (ndarray): Test kernel matrix.
        
        Returns:
            out (ndarray): Predicted probabilities.
        """
        f = np.dot(K_test, self.alpha)
        p = 1 / (1 + np.exp(-f))
        return p

    def predict(self, K_test):
        """
        Predicts class labels for test data.
        
        Parameters:
            K_test (ndarray): Test kernel matrix.
        
        Returns:
            out (ndarray): Predicted labels (0 or 1).
        """
        proba = self.predict_proba(K_test)
        return (proba >= 0.5).astype(int)

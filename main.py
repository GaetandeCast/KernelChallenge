import numpy as np
import pandas as pd

from kernel_logistic_regression import KernelLogisticRegression
from kernel_svc import KernelSVC
from kernels import spectrum_kernel


# Parameters
C = 1
kmer_size = 6
mismatch_allowed = None
epsilon = 1e-9

def evaluate():    

    #scores_SVC = []
    #scores_LRR = []
    lambdas = [0.01, 0.1, 1, 10, 100, 1000]
    scores_LRR = np.zeros(len(lambdas))

    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    scores_SVC = np.zeros(len(Cs))

    for k in [1]:
        # Construct file names
        train_seq_file = f"data/Xtr{k}.csv"
        train_label_file = f"data/Ytr{k}.csv"
        
        # Read the CSV files 
        X = pd.read_csv(train_seq_file, header=0).iloc[:, 1].tolist()
        y = pd.read_csv(train_label_file, header=0).iloc[:, 1].tolist()
        
        # Split the data into training and validation sets
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X))
        X = [X[i] for i in perm]
        y = [y[i] for i in perm]
        n = len(X)
        n_train = int(0.8 * n)
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        print("Computing kernel matrices...")
        kernel = spectrum_kernel(kmer_size=kmer_size, mismatch=mismatch_allowed)
        X_train_fvecs = [kernel.get_feature_vector(seq) for seq in X_train]
        X_val_fvecs = [kernel.get_feature_vector(seq) for seq in X_val]
        K_train = kernel.compute_kernel_matrix(X_train_fvecs, X_train_fvecs)
        K_val = kernel.compute_kernel_matrix(X_val_fvecs, X_train_fvecs)

    
        """
        for (i,C) in enumerate(Cs):
            model = KernelSVC(C=C, epsilon=epsilon)
            print(f"Training SVC for dataset {k} with {len(X_train)} sequences...")
            model.fit(K=K_train, vectors=X_train_fvecs, y=y_train)
            
            pred = (model.predict(K_val)>0).astype(int) # Convert back to 0, 1
            score = np.mean(pred == y_val)
            scores_SVC[i] += score
            print(f"Validation score for C= {C}: {score:.4f}")
        
        """
        for (i,reg) in enumerate(lambdas):
            model = KernelLogisticRegression(n_iter=50, tol=epsilon, reg=reg)
            #print(f"Training LRR for dataset {k} with {len(X_train)} sequences...")
            model.fit(K=K_train, y=y_train)
            
            pred = (model.predict(K_val)>0).astype(int) # Convert back to 0, 1
            score = np.mean(pred == y_val)
            scores_LRR[i] += score
            print(f"Validation score for lamda= {lambdas[i]}: {score:.4f}")
        
    print([f"Lambda : {lambdas[i]}, mean score: {scores_LRR[i]/3}" for i in range(len(lambdas))])
    #print(f"Validation scores SVC, LRR: {scores_SVC, scores_LRR}")
    #print(f"Mean validation score: {[np.mean(scores_SVC), np.mean(scores_LRR)]:.4f}")


def predict():

    submission_list = []
    kmers = [3,6,3]

    for k in [0, 1, 2]:
        # Construct file names
        train_seq_file = f"data/Xtr{k}.csv"
        train_label_file = f"data/Ytr{k}.csv"
        test_seq_file = f"data/Xte{k}.csv"
        
        # Read the CSV files 
        X_train = pd.read_csv(train_seq_file, header=0).iloc[:, 1].tolist()
        y_train = pd.read_csv(train_label_file, header=0).iloc[:, 1].tolist()
        X_test = pd.read_csv(test_seq_file, header=0).iloc[:, 1].tolist()
        

        print("Computing kernel matrices...")
        kernel = spectrum_kernel(kmer_size=kmers[k], mismatch=None)

        X_train_fvecs = [kernel.get_feature_vector(seq) for seq in X_train]
        K_train = kernel.compute_kernel_matrix(X_train_fvecs, X_train_fvecs)

        X_test_fvecs = [kernel.get_feature_vector(seq) for seq in X_test]
        K_test = kernel.compute_kernel_matrix(X_test_fvecs, X_train_fvecs)

        # Create and train the model
        model = KernelLogisticRegression(n_iter=50, tol=epsilon, reg=100)
        print(f"Training LRR for dataset {k} with {len(X_train)} sequences...")
        model.fit(K=K_train, y=y_train)

        # Predict on test sequences
        train_pred = (model.predict(K_train)>0).astype(int) # Convert back to 0, 1
        print(f"Training accuracy: {np.mean(train_pred == y_train):.4f}")
        test_preds = (model.predict(K_test)>0).astype(int) # Convert back to 0, 1
        
        # Create an Id column for test sequences
        ids = np.arange(1000 * k, 1000 * k + len(X_test))
        df = pd.DataFrame({'Id': ids, 'Bound': test_preds})
        submission_list.append(df)

    # Concatenate predictions from all datasets and save as a single CSV file.
    submission = pd.concat(submission_list, ignore_index=True)
    submission.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' created.")

if __name__ == "__main__":
    #evaluate()
    predict()
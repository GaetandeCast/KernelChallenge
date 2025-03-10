import numpy as np
import pandas as pd

from kernel_logistic_regression import KernelLogisticRegression
# Parameters
kernel_type = 'spectrum'      
kmer_size = 3             
mismatch_allowed = 1      
n_iter = 100              
tol = 1e-6
reg = 0.01

from kernel_svc import KernelSVC
# Parameters
C = 1
kernel_type = 'spectrum'
kmer_size = 3
mismatch_allowed = 1
epsilon = 1e-9

submission_list = []

for k in [0, 1, 2]:
    # Construct file names
    train_seq_file = f"data/Xtr{k}.csv"
    train_label_file = f"data/Ytr{k}.csv"
    test_seq_file = f"data/Xte{k}.csv"
    
    # Read the CSV files 
    X_train = pd.read_csv(train_seq_file, header=0).iloc[:, 1].tolist()
    y_train = pd.read_csv(train_label_file, header=0).iloc[:, 1].tolist()
    X_test = pd.read_csv(test_seq_file, header=0).iloc[:, 1].tolist()
    
    # Create and train the model
    #model = KernelLogisticRegression(kernel=kernel_type, kmer_size=kmer_size, mismatch=mismatch_allowed, n_iter=n_iter, tol=tol, reg=reg)
    model = KernelSVC(C=C, kernel=kernel_type, kmer_size=kmer_size, mismatch=mismatch_allowed, epsilon=epsilon)
    print(f"Training model for dataset {k} with {len(X_train)} sequences...")
    model.fit(X_train, y_train)
    
    # Predict on test sequences
    train_pred = model.predict(X_train)>0 # Convert back to 0, 1
    print(f"Training accuracy: {np.mean(train_pred == y_train):.4f}")
    test_preds = model.predict(X_test)>0 # Convert back to 0, 1
    
    # Create an Id column for test sequences
    ids = np.arange(1000 * k, 1000 * k + len(X_test))
    df = pd.DataFrame({'Id': ids, 'Bound': test_preds})
    submission_list.append(df)

# Concatenate predictions from all datasets and save as a single CSV file.
submission = pd.concat(submission_list, ignore_index=True)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")

import numpy as np
import pandas as pd

def load_data(csv_path: str, target_col: str, train_ratio: float = 0.8, seed: int = 42):
    # load data
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # bo cot khong phai so (id, date, ...)
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    df = df.drop(columns=non_numeric)

    # chia tap X va Y
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    Y = df[[target_col]].to_numpy(dtype=float)

    # Chia data
    n_samples  = len(X)
    n_train = int(n_samples*train_ratio)

    np.random.seed(seed)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    # chuan hoa data

    mean_X = np.mean(X_train, axis=0, keepdims=True)
    std_X = np.std(X_train, axis=0, keepdims=True)
    std_X[std_X == 0] = 1.0

    X_train = (X_train - mean_X)/std_X
    X_test = (X_test - mean_X)/std_X

    features_names = [col for col in df.columns if col != target_col]

    return X_train, Y_train, X_test, Y_test, features_names, (mean_X, std_X)

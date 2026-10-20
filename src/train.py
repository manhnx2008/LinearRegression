import numpy as np
from src.model import LinearRegression

def train(model: LinearRegression, X_train: np.ndarray, Y_train: np.ndarray, epochs: int = 1000, lr = 0.01, tol:float=1e-5):
    N,D = X_train.shape

    model.w = np.zeros((D,1), dtype=float)
    model.b = 0.0

    pre_loss = float('inf')

    for epoch in range(epochs):
        Y_predict = model.predict(X_train)

        loss = np.mean((Y_predict-Y_train)**2)/2.0

        if abs(pre_loss - loss) < tol:
            break

        pre_loss = loss
        error = Y_predict - Y_train

        dw = (1/N)*(X_train.T@error)
        db = (1/N)*np.sum(error)

        model.w -= lr*dw
        model.b -= lr*db

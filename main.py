from data.data import load_data
from src.model import LinearRegression
from src.train import train

def main():
    X_train, Y_train, X_test, Y_test, feature_names, (mean_X, std_X) = load_data(csv_path='data/Housing.csv', target_col='price')
    model = LinearRegression()

    train(model, X_train, Y_train)

    Y_predict = model.predict(X_test[:100])
    for i in range(len(Y_predict)):
        print(f"Predicted: {Y_predict[i][0]:.2f}  |  Actual: {Y_test[i][0]:.2f}")

if __name__ == '__main__':
    main()


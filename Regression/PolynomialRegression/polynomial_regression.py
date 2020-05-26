import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Step # 1: Data Prepossessing
    # Reading the data
    print("Reading the data...")
    dataset = pd.read_csv("../../DataSets/PolynomialRegression/Position_Salaries.csv")
    print("Dataset contains")
    print(dataset.count())

    # Extracting features and outcome from dataset
    print("Extracting features and outcome from dataset...")
    feature = dataset.iloc[:, 1:-1].values
    outcome = dataset.iloc[:, -1].values

    # Converting the feature into polynomial form
    print("Converting the feature into polynomial form...")
    from sklearn.preprocessing import PolynomialFeatures
    polynomial_features = PolynomialFeatures(degree=4)
    polynomial_feature = polynomial_features.fit_transform(feature)

    # Step # 2: Training the model
    print("Training the model")
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(polynomial_feature, outcome)

    # Increase the resolution of the feature that is going to use in prediction
    resolute_feature = np.arange(min(feature), max(feature), 0.001)
    resolute_feature = resolute_feature.reshape((len(resolute_feature), 1))

    # Predicting the outcome and plotting the graph
    plt.scatter(feature, outcome, color='red')
    plt.plot(resolute_feature, regressor.predict(polynomial_features.fit_transform(resolute_feature)), color='black')
    plt.show()

    # Predicting the salary of an employee having X years of experience
    salary_to_predict = np.array(6.5, dtype=float).reshape(1, -1)
    print("The predicted salary is", end=' : ')
    print(regressor.predict(polynomial_features.fit_transform(salary_to_predict)))


if __name__ == '__main__':
    main()

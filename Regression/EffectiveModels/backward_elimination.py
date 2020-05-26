import pandas as pd
import matplotlib.pyplot as plt


def main():

    # Step # 1: Data Prepossessing
    # Reading the data
    print("Reading the data...")
    dataset = pd.read_csv('../../DataSets/MultipleLinearRegression/50_Startups.csv')
    print("Dataset contains")
    print(dataset.count())

    # Extracting features and outcome from dataset
    print("Extracting features and outcome from dataset...")
    features = dataset.iloc[:, :-1].to_numpy()
    outcome = dataset.iloc[:, -1].to_numpy()

    # Handling the categorical data
    print('Handling the categorical data')
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    column_transformer = ColumnTransformer([('DummyVariables', OneHotEncoder(), [3])], remainder='passthrough')
    features = column_transformer.fit_transform(features)

    # Avoiding the Dummy Variable trap - scikit-learn handles the dummy variable trap, but just for demo
    # I'm removing ane dummy column.
    print('Avoiding Dummy Variable trap')
    features = features[:, 1:]

    # Building the optimal model by using backward elimination
    # Adding the constant in the features matrix
    import statsmodels.api as sm
    features = sm.add_constant(features)

    optimal_features = features[:, [0, 1, 2, 3, 4, 5]]
    optimal_features = optimal_features.astype('float64')

    optimal_features = features[:, [0, 1, 3, 4, 5]]
    optimal_features = optimal_features.astype('float64')

    optimal_features = features[:, [0, 3, 4, 5]]
    optimal_features = optimal_features.astype('float64')

    optimal_features = features[:, [0, 3, 5]]
    optimal_features = optimal_features.astype('float64')

    optimal_features = features[:, [0, 3]]
    optimal_features = optimal_features.astype('float64')

    # Training the model
    ols_regressor = sm.OLS(endog=outcome, exog=optimal_features).fit()

    print(ols_regressor.summary())

    print(ols_regressor.predict())

    plt.scatter(optimal_features[:, 0], outcome, color='red')
    plt.plot(optimal_features[:, 0], ols_regressor.predict(), color='black')
    plt.xlabel("Constant")
    plt.ylabel("Profit")
    plt.title("Effective Linear Regression Model")
    plt.show()

    plt.scatter(optimal_features[:, 1], outcome, color='red')
    plt.plot(optimal_features[:, 1], ols_regressor.predict(), color='blue')
    plt.xlabel("R&D")
    plt.ylabel("Profit")
    plt.title("Effective Linear Regression Model")
    plt.show()


if __name__ == '__main__':
    main()

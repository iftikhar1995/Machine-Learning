import pandas as pd


def main():
    # Step # 1: Data Prepossessing
    # Reading the data
    print("Reading the data...")
    dataset = pd.read_csv('../../DataSets/MultipleLinearRegression/50_Startups.csv')
    print("Dataset contains")
    print(dataset.count())

    # Extracting features and outcome from dataset
    print("Extracting features and outcome from dataset...")
    features = dataset.iloc[:, :-1].values
    outcome = dataset.iloc[:, -1].values

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

    # Splitting the data into two parts for testing and training purposes
    from sklearn.model_selection import train_test_split
    print('Splitting the data into two parts for testing and training purposes')
    features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.2,
                                                                                  random_state=0)
    print("Training dataset contains {0} records".format(len(features_train)))
    print("Test dataset contains {0} records".format(len(features_test)))

    # We don't have to apply feature scaling here as scikit-learn handles that for us.

    # Step # 2: Training the model
    print('Training the model')
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(features_train, outcome_train)

    # Step # 3: Making Predictions
    print('Making Predictions')
    predicted_outcome = linear_regressor.predict(features_test)

    print(predicted_outcome)
    print(outcome_test)


if __name__ == '__main__':
    main()

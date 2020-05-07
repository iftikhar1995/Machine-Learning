import numpy as np
import matplotlib as plt
import pandas as pd


def main():

    # Step-1 :: Importing the dataset
    dataset = pd.read_csv("../DataSets/DataPreprocessing/Data.csv")

    # Step-2 :: Separating the features and the outcome
    features = dataset.iloc[:, :-1].values
    outcome = dataset.iloc[:, -1].values

    # Step-3 :: Handling the missing data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(X=features[:, 1:3])
    features[:, 1:3] = imputer.transform(X=features[:, 1:3])

    # Handling the Categorical Data
    # Introducing dummy columns
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    column_transformer = ColumnTransformer([("dummy_cols", OneHotEncoder(), [0])], remainder="passthrough")
    features = column_transformer.fit_transform(features)

    # encoding dependent variable
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    outcome = label_encoder.fit_transform(y=outcome)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.2,
                                                                                  random_state=0)
    # Applying feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_features = StandardScaler()
    features_train = sc_features.fit_transform(features_train)
    features_test = sc_features.transform(features_test)

    print("*" * 25, "features_train", "*" * 25)
    print(features_train)

    print("*" * 25, "features_test", "*" * 25)
    print(features_test)

    print("*" * 25, " outcome_train", "*" * 25)
    print(outcome_train)

    print("*" * 25, " outcome_test", "*" * 25)
    print(outcome_test)


if __name__ == '__main__':
    main()

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Step 1 :: Prepossess the data
    # Reading the dataset
    print("Reading the Dataset")
    dataset = pd.read_csv('../../DataSets/SimpleLinearRegression/height_and_weight_data.csv')
    print("Dataset contains {0} records".format(dataset.count()))

    # Extracting height(independent variable) and weight(dependent variable)
    # from dataset
    print("Extracting height and weight from the dataset")
    height = dataset.iloc[:, :-1].values
    weight = dataset.iloc[:, -1].values

    # Splitting dataset for training and testing purposes
    print("Splitting dataset for training and testing purposes")
    from sklearn.model_selection import train_test_split
    height_train, height_test, weight_train, weight_test = train_test_split(height, weight, test_size=1/3,
                                                                            random_state=0)
    print("Training dataset contains {0} records".format(len(height_train)))
    print("Test dataset contains {0} records".format(len(height_test)))

    # Step 2 :: Training model on training set
    print("Fitting linear regression model to dataset")
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X=height_train, y=weight_train)

    # Step 3 :: Predicting the weight of the person
    print("Predicting the weight")
    predicted_weight = linear_regression.predict(X=height_test)

    # Step 4 :: Visualizing the training set
    print("Plotting Weight vs Height graph on training dataset")
    plt.scatter(x=height_train, y=weight_train, color='#6f0000')
    plt.plot(height_train, linear_regression.predict(height_train), color='#000000')
    plt.title("Weight vs Height <Training Data>")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.show()

    # Step 4 :: Visualizing the Test set
    print("Plotting Weight vs Height graph on Test dataset")
    plt.scatter(x=height_test, y=weight_test, color='#6f0000')
    plt.plot(height_test, linear_regression.predict(height_test), color='#000000')
    plt.title("Weight vs Height <Test Data>")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.show()


if __name__ == '__main__':
    main()

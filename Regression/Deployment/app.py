import pandas as pd
import numpy as np
import pickle


from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/')
def main():
    return 'Container is up & running'


@app.route('/SimpleLinearRegression/Train')
def train():
    # Step 1 :: Prepossess the data
    # Reading the dataset
    print("Reading the Dataset")
    dataset = pd.read_csv('height_and_weight_data.csv')
    print("Dataset contains {0} records".format(dataset.count()))

    # Extracting height(independent variable) and weight(dependent variable)
    # from dataset
    print("Extracting height and weight from the dataset")
    height = dataset.iloc[:, :-1].values
    weight = dataset.iloc[:, -1].values

    # Splitting dataset for training and testing purposes
    print("Splitting dataset for training and testing purposes")
    from sklearn.model_selection import train_test_split
    height_train, height_test, weight_train, weight_test = train_test_split(height, weight, test_size=1 / 3,
                                                                            random_state=0)
    print("Training dataset contains {0} records".format(len(height_train)))
    print("Test dataset contains {0} records".format(len(height_test)))

    # Step 2 :: Training model on training set
    print("Fitting linear regression model to dataset")
    from sklearn.linear_model import LinearRegression
    linear_regression = LinearRegression()
    linear_regression.fit(X=height_train, y=weight_train)

    with open("linear_regression.pickle","wb") as pickle_out:
        pickle.dump(linear_regression, pickle_out)

    return "Train Successfully"


@app.route('/SimpleLinearRegression/Predict')
def predict():

    height = request.args.get('height')
    _input = np.array([height], dtype=float).reshape(-1, 1)

    with open("linear_regression.pickle", "rb") as pickle_in:
        linear_regression = pickle.load(pickle_in)
        result = linear_regression.predict(_input)
        return "The predict weight against height {0} is {1}".format(height, round(result[0], 2))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

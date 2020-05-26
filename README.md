# Machine Learning
The repository contains code for different types of Machine Learning Models. For each type of Model, I have performed 
following steps:
1. Preprocess the data
2. Training model
3. Making Predictions
4. Visualizing the predictions on training and test data

Along with this I have added a data prepossessing script that can be used in most of the cases.
# Tools and Technologies:

Following are the tools and technologies used in the project:
- Scikit-Learn
- Pandas
- Matplotlib
- Python

 # Project Structure

```
    Machine Learning
        |
        |- DataPreprocessing
        |   |
        |   |- data_preprocessing.py
        |
        |- DataSets
        |   |
        |   |- DataPreprocessing
        |   |       |- Data.csv
        |   |- MultipleLinearRegression
        |   |       |- 50_Startups.csv
        |   |- PolynomialRegression
        |   |       |- Position_Salaries.csv
        |   |- SimpleLinearRegression
        |   |       | - height_and_weight_data.csv
        |   |
        |
        |- Regression
        |   |
        |   | - EffectiveModels
        |   |       |- backward_elimination.py
        |   | - MultipleLinearRegression
        |   |       |- multiple_linear_regression.py
        |   | - PolynomialRegression
        |   |       |- polynomial_regression.py
        |   | - SimpleLinearRegression
        |   |       |- simple_linear_regression.py
        |   |
        |
        |- .gitignore
        |- requirements.txt
```

# Getting Started

Following instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

1. Clone the repository using below command:\
   ```git clone <https://github.com/iftikhar1995/Machine-Learning.git>```

2. Install the dependencies mentioned in the **requirements.txt** file. Following is the command to install the 
dependencies:\
```pip install -r requirements.txt ```

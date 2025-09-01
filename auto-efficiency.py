import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep = r'\s+', header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data.drop('car name', inplace = True, axis = 1) #car name is of no use to our predictions
#dropping the rows with ? in horsepower since they are very few and would be a problem for fitting our tree
data['horsepower'] = data['horsepower'].replace('?', np.nan)
data = data.dropna()
#since origin is a categorical feature, we one hot encode it
data = pd.get_dummies(data, columns = ['origin'], prefix = ['origin'])
#defining the feature-data and the output or target variable
y = data['mpg']
X = data.drop('mpg', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

#My Decision Tree
my_tree = DecisionTree(criterion = 'gini_index', max_depth = 5) #because sklearn uses gini index in its DecisionTreeRegressor
my_tree.fit(X_train, y_train)
my_y_hat = my_tree.predict(X_test)
my_rmse = rmse(my_y_hat, y_test)
print(f"My Decision Tree's RMSE = {my_rmse:.2f}")

#Scikitlearn's Decision Tree
sklearn_tree = DecisionTreeRegressor(max_depth = 5, random_state = 7)
sklearn_tree.fit(X_train, y_train)
sklearn_y_hat = sklearn_tree.predict(X_test)
sklearn_rmse = np.sqrt(mean_squared_error(sklearn_y_hat, y_test))
print(f"Scikitlearn's Decision Tree's RMSE = {sklearn_rmse:.2f}")

#output:
"""
My Decision Tree's RMSE = 3.59
Scikitlearn's Decision Tree's RMSE = 3.90
"""

"""
My tree might be working better than Scikit because scikit's implementation is more complex and optimized for speed.
And it did take a significantly less time to run than my model. The dataset being small might have been the reason for
my model performing slightly better
"""

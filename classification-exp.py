import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

#Q2 a)

X_df = pd.DataFrame(X)
y_s = pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size = 0.3, random_state = 7)
tree = DecisionTree(criterion = "information_gain", max_depth = 5)
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
print("Results for Part(a)")
print(f"Accuracy : {accuracy(y_hat, y_test):.2f}")
for cls in y_test.unique():
    p = precision(y_hat, y_test, cls)
    r = recall(y_hat, y_test, cls)
    print(f"Class {cls} - Precision: {p:.2f}, Recall: {r:.2f}") 


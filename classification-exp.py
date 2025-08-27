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
#sample results:
""" 
Results for Part(a)
Accuracy : 0.87
Class 0 - Precision: 0.81, Recall: 0.93
Class 1 - Precision: 0.93, Recall: 0.81 
"""

#Q2 b)

from sklearn.model_selection import KFold

possible_depths = list(range(2, 8))
outer_kfold= KFold(n_splits = 5, shuffle = True, random_state = 7)
outer_loop_scores = []
print("Results for part (b)")
for train_index, test_index in outer_kfold.split(X_df):
    X_outer_train, X_outer_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_outer_train, y_outer_test = y_s.iloc[train_index], y_s.iloc[test_index]
    best_depth = -1
    best_avg_acc = -1.0
    for depth in possible_depths:
        inner_kfold = KFold(n_splits = 3, shuffle = True, random_state = 7)
        sum_inner_loop_scores = 0
        for inner_train_index, inner_eval_index in inner_kfold.split(X_outer_train):
            X_inner_train, X_inner_eval = X_outer_train.iloc[inner_train_index], X_outer_train.iloc[inner_eval_index]
            y_inner_train, y_inner_eval = y_outer_train.iloc[inner_train_index], y_outer_train.iloc[inner_eval_index]
            tree = DecisionTree(criterion = "information_gain", max_depth = depth)
            tree.fit(X_inner_train, y_inner_train)
            y_hat = tree.predict(X_inner_eval)
            sum_inner_loop_scores += accuracy(y_hat, y_inner_eval)
        avg_accuracy_curr_depth = sum_inner_loop_scores / 3
        if  avg_accuracy_curr_depth > best_avg_acc:
            best_avg_acc = avg_accuracy_curr_depth
            best_depth = depth
    final_tree = DecisionTree(criterion = "information_gain", max_depth = best_depth)
    final_tree.fit(X_outer_train, y_outer_train)
    y_final_hat = final_tree.predict(X_outer_test)
    outer_score = accuracy(y_final_hat, y_outer_test)
    outer_loop_scores.append(outer_score)
mean_acc = np.mean(outer_loop_scores)
std_acc = np.std(outer_loop_scores)
print(f"Average accuracy across 5 folds : {mean_acc:.2f}")
print(f"Standard Deviation of Accuracy: {std_acc:.2f}")

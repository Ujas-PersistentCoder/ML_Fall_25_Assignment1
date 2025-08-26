"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self, value = None):
        self.value = value #for leaf node
        #for decision nodes
        self.feature = None
        self.isreal_feature = False
        self.threshold = None
        self.left_child = None
        self.right_child = None
        
@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def _grow_tree(self, X, y, depth = 0): #helper function, only for interanl function
        if (depth >= self.max_depth or len(y.unique())==1 or X.shape[0] < 2):
            output_isreal = check_ifreal(y)
            if output_isreal:
                leaf_value = y.mean()
            else: 
                leaf_value = y.mode()[0]
            return Node(value = leaf_value)
        else:
            features = list(X.columns)
            best_feature, best_value = opt_split_attribute(X, y, self.criterion, features)
            current_node = Node()
            current_node.feature = best_feature
            current_node.threshold = best_value
            current_node.is_real_feature = check_ifreal(X[best_feature])
            X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_value)
            current_node.left_child = self._grow_tree(X_left, y_left, depth+1)
            current_node.right_child = self._grow_tree(X_right, y_right, depth+1)
            return current_node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.root = self._grow_tree(X, y)

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

    def _traverse_tree(self, row, node):
        if (node.value is not None):
            return node.value;
        feature_value = row[node.feature]
        if node.is_real_feature:
            if feature_value <= node.threshold:
                return self._traverse_tree(row, node.left_child)
            return self._traverse_tree(row, node.right_child)
        else:
            if feature_value == node.threshold:
                return self._traverse_tree(row, node.left_child)
            return self._traverse_tree(row, node.right_child)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = X.apply(lambda row : self._traverse_tree(row, self.root), axis = 1)
        return predictions

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

    def _plot_recursive(self, node, indent = ""):
        if node.value is not None:
            print(indent + "--> Predict:", node.value)
            return
        if node.isreal_feature:
            question = f"? ({node.feature} <= {node.threshold})"
        else:
            question = f"? ({node.feature} == {node.threshold})"
        print(indent + question)
        print (indent + "Y:")
        self._plot_recursive(node.left_child, indent + "    ")
        print(indent + "N:")
        self._plot_recursive(node.right_child, indent + "    ")
        
    def plot(self) -> None:
        self._plot_recursive(self.root)
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass

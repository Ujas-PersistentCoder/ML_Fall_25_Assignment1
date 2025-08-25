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
        if (depth >= self.max_depth or len(y.unique())==1 or len(X) < 2):
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
            X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_value)
            current_node.left_child = self._grow_tree(X_left, y_left, depth+1)
            current_node.right_child = self._grow_tree(X_right, y_right, depth+1)
            return current_node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

    def plot(self) -> None:
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

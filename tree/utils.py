"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np #included numpy to use functions like log
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    pass


def entropy(Y: pd.Series) -> float:
    class_counts = Y.value_counts()
    probs = class_counts / len(Y)
    entropy_value = (-probs * np.log2(probs)).sum()
    return entropy_value

#Though gini_index and entropy serve the same purpose and would mostly give the same splits, gini index is faster due to the abscence of log and works on just simple squaring
#The Gini Index essentially calculates the probability of misclassifying a randomly chosen marble
#Gini Index is often the default criterion in popular libraries like scikit-learn

def gini_index(Y: pd.Series) -> float:
    class_counts = Y.value_counts()
    probs = class_counts / len(Y)
    gini_value = 1 - (probs ** 2).sum()
    return gini_value

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    pass


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass

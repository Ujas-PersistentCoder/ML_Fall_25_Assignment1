"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np #included numpy to use functions like log
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    threshold = 10
    if not pd.api.types.is_numeric_dtype(y): return False
    else :
        if (y.nunique() > threshold): return True
        else return False

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
    is_output_real = check_ifreal(Y)
    if (is_output_real):
        y_mean = Y.mean()
        initial_mse = ((Y - y_mean) ** 2).mean()
        weighted_mse = 0.0
        unique_values = attr.unique()
        for value in unique_values:
            child_Y = Y[attr == value]
            if len(child_Y):
                weight = len(child_Y) / len(Y)
                child_mean = child_Y.mean()
                child_mse = ((child_Y - child_mean) ** 2).mean()
                weighted_mse += weight * child_mse
        return initial_mse - weighted_mse
    else:
        initial_impurity = gini_index(Y) if (criterion == 'gini_index') else entropy(Y) 
        weighted_impurity = 0.0
        class_values = attr.unique()
        for value in class_values:
            child_Y = Y[attr == value]
            weight = len(child_Y) / len(Y)
            child_impurity = gini_index(child_Y) if (criterion == 'gini_index') else entropy(child_Y)
            weighted_impurity += child_impurity * weight
        info_gain = initial_impurity - weighted_impurity
        return info_gain
    
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    max_info_gain = -1.0
    best_attr = None
    best_split_value = None
    for attr in features:
        is_real = check_ifreal(X[attr])
        if is_real:
            unique_values = sorted(X[attr].unique())
            best_ig_attr = -1.0
            best_threshold_attr = None
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                temp_split = X[attr] <= threshold
                curr_ig = information_gain(y, temp_split, criterion)
                if curr_ig > best_ig_attr:
                    best_ig_attr = curr_ig
                    best_threshold_attr = threshold
            if best_ig_attr > max_info_gain:
                best_attr = attr
                best_split_value = best_threshold_attr
                max_info_gain = best_ig_attr
        else:
            info_gain = information_gain(y, X[attr], criterion)
            if (info_gain > max_info_gain):
                max_info_gain = info_gain
                best_attr = attr
                best_split_value = None
    return best_attr, best_split_value
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    attr_isreal = check_ifreal(X[attribute])
    if attr_isreal:
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value
    X_left= X[left_mask]
    y_left = y[left_mask]
    X_right = X[right_mask]
    y_right = y[right_mask]
    return X_left, y_left, X_right, y_right
            
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

from typing import Union
import pandas as pd
import numpy as np

"""Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases."""

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    The following assert checks if sizes of y_hat and y are equal.
    """
    assert y_hat.size == y.size
    return (y_hat == y).sum() / y.count()


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    if ((y_hat == cls).sum() == 0): return 0.0
    return ((y_hat == cls) & (y == cls)).sum() / (y_hat == cls).sum()


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    if ((y == cls).sum() == 0): return 0.0
    return ((y_hat == cls) & (y == cls)).sum() / (y == cls).sum()

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return np.sqrt(((y_hat - y) ** 2).mean())


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return (np.abs(y_hat - y)).mean()

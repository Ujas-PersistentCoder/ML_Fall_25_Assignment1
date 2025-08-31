import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
def generate_fake_data(n, m, input_type, output_type):
  if (input_type == 'discrete'):
    X = pd.DataFrame(np.random.randint(2, size = (n, m)))
  else:
    X = pd.DataFrame(np.random.randn(n, m))
  if (output_type == 'discrete'):
    y = pd.Series(np.random.randint(2, size = n), dtype = "category")
  else:
    y = pd.Series(np.random.randn(n))
  return X, y

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def run_time_experiment(N_values, M_values, input_type, output_type, n):
  avg_training_times_N, avg_prediction_times_N = [], []
  avg_training_times_M, avg_prediction_times_M = [], []
  M_fixed = 15
  for N in N_values:
    temp_train_times_sum, temp_predict_times_sum = 0, 0
    for i in range(n):
      X, y = generate_fake_data(N, M_fixed, input_type, output_type)
      tree = DecisionTree(criterion = "information_gain", max_depth = 5)
      start = time.time()
      tree.fit(X, y)
      end = time.time()
      temp_train_times_sum += end - start
      start = time.time()
      tree.predict(X)
      end = time.time()
      temp_predict_times_sum += end - start
    avg_training_times_N.append(temp_train_times_sum / n)
    avg_prediction_times_N.append(temp_predict_times_sum / n)
  N_fixed = 500
  for M in M_values:
    temp_train_times_sum, temp_predict_times_sum = 0, 0
    for i in range(n):
      X, y = generate_fake_data(N_fixed, M, input_type, output_type)
      tree = DecisionTree(criterion = "information_gain", max_depth = 5)
      start = time.time()
      tree.fit(X, y)
      end = time.time()
      temp_train_times_sum += end - start
      start = time.time()
      tree.predict(X)
      end = time.time()
      temp_predict_times_sum += end - start
    avg_training_times_M.append(temp_train_times_sum / n)
    avg_prediction_times_M.append(temp_predict_times_sum / n)
  return {
        "train_N": avg_training_times_N, "pred_N": avg_prediction_times_N,
        "train_M": avg_training_times_M, "pred_M": avg_prediction_times_M,
        "N_fixed": N_fixed, "M_fixed": M_fixed
    }
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

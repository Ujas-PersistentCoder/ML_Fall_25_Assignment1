import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(7)
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
      tree = DecisionTree(criterion = "entropy", max_depth = 5)
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
      tree = DecisionTree(criterion = "entropy", max_depth = 5)
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
N_values = [50, 100, 200, 400, 800, 1600]
M_values = [5, 10, 15, 20, 25, 30]
cases = {
    "Discrete In, Discrete Out": ("discrete", "discrete"),
    "Real In, Discrete Out": ("real", "discrete"),
    "Discrete In, Real Out": ("discrete", "real"),
    "Real In, Real Out": ("real", "real")
}

results = {}
print("--- Running Runtime Complexity Experiments ---")
for case_name, (in_type, out_type) in cases.items():
    print(f"Running case: {case_name}...")
    results[case_name] = run_time_experiment(N_values, M_values, in_type, out_type, num_average_time)
print("Experiments complete. Generating plots.")

# --- Plotting the Results ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
fig.suptitle('Training Time Complexity Analysis', fontsize=16)
plot_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (case_name, data), (r, c) in zip(results.items(), plot_coords):
    ax = axes[r, c]
    # Plot training time vs N
    ax.plot(N_values, data["train_N"], marker='o', linestyle='-', label=f'Varying N (M={data["M_fixed"]})')
    # Plot training time vs M on a secondary y-axis if scales differ, or just plot
    # For simplicity, we plot them on the same axis but this can be misleading if scales differ.
    # A better approach is often two separate plots per case.
    ax2 = ax.twiny() # Create a second x-axis
    ax2.plot(M_values, data["train_M"], marker='x', linestyle='--', color='r', label=f'Varying M (N={data["N_fixed"]})')
    
    ax.set_title(case_name)
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Time (seconds)")
    ax2.set_xlabel("Number of Features (M)")
    
    # Manually create legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Asst_RuntimeAnalysis_Q4.png")
plt.show()

# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

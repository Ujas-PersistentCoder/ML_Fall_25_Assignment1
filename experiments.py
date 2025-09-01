import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(7)
num_average_time = 100  # Reduced for faster testing, you can increase it back to 100

# Function to create fake data
def generate_fake_data(n, m, input_type, output_type):
    if (input_type == 'discrete'):
        X = pd.DataFrame(np.random.randint(2, size=(n, m)))
    else:
        X = pd.DataFrame(np.random.randn(n, m))
    if (output_type == 'discrete'):
        y = pd.Series(np.random.randint(2, size=n), dtype="category")
    else:
        y = pd.Series(np.random.randn(n))
    return X, y

# Function to calculate average time taken by fit() and predict()
def run_time_experiment(N_values, M_values, input_type, output_type, n_runs):
    avg_training_times_N, avg_prediction_times_N = [], []
    avg_training_times_M, avg_prediction_times_M = [], []
    
    # --- Vary N ---
    M_fixed = 10
    total_N = len(N_values)
    # Use enumerate to get the iteration number (i)
    for i, N in enumerate(N_values):
        # Add this print statement for progress
        print(f"  Varying N: Processing size {i+1}/{total_N} (N={N})")
        
        temp_train_times, temp_predict_times = [], []
        for _ in range(n_runs):
            X, y = generate_fake_data(N, M_fixed, input_type, output_type)
            tree = DecisionTree(criterion="entropy", max_depth=5)
            
            start = time.time()
            tree.fit(X, y)
            temp_train_times.append(time.time() - start)

            start = time.time()
            tree.predict(X)
            temp_predict_times.append(time.time() - start)
            
        avg_training_times_N.append(np.mean(temp_train_times))
        avg_prediction_times_N.append(np.mean(temp_predict_times))

    # --- Vary M ---
    N_fixed = 50
    total_M = len(M_values)
    # Use enumerate to get the iteration number (i)
    for i, M in enumerate(M_values):
        # Add this print statement for progress
        print(f"  Varying M: Processing size {i+1}/{total_M} (M={M})")
        
        temp_train_times, temp_predict_times = [], []
        for _ in range(n_runs):
            X, y = generate_fake_data(N_fixed, M, input_type, output_type)
            tree = DecisionTree(criterion="entropy", max_depth=5)

            start = time.time()
            tree.fit(X, y)
            temp_train_times.append(time.time() - start)
            
            start = time.time()
            tree.predict(X)
            temp_predict_times.append(time.time() - start)
            
        avg_training_times_M.append(np.mean(temp_train_times))
        avg_prediction_times_M.append(np.mean(temp_predict_times))
        
    return {
        "train_N": avg_training_times_N, "pred_N": avg_prediction_times_N,
        "train_M": avg_training_times_M, "pred_M": avg_prediction_times_M,
        "N_fixed": N_fixed, "M_fixed": M_fixed
    }

# --- Main Script ---
N_values = [10, 20, 40, 80, 160]
M_values = [2, 5, 10, 15, 20]
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
    # Create a second x-axis for varying M
    ax2 = ax.twiny()
    ax2.plot(M_values, data["train_M"], marker='x', linestyle='--', color='r', label=f'Varying M (N={data["N_fixed"]})')
    
    ax.set_title(case_name)
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Time (seconds)")
    ax2.set_xlabel("Number of Features (M)")
    
    # Manually create combined legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Asst_RuntimeAnalysis_Q4.png")
plt.show()

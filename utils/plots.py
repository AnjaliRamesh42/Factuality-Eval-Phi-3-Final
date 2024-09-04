import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_variances(variance_files):
    variances = {}
    for file in variance_files:
        # Extract the sampling method name from the filename
        sampling_method = os.path.splitext(os.path.basename(file))[0].split("_", 1)[1]
        with open(file, 'r') as f:
            variances[sampling_method] = json.load(f)
    return variances

def plot_variances(variances):
    metrics = list(next(iter(variances.values())).keys())  # Get the metrics from the first sampling method
    sampling_methods = list(variances.keys())
    
    # Set up the bar width and positions
    bar_width = 0.2
    r = np.arange(len(metrics))
    
    plt.figure(figsize=(12, 6))
    
    for i, method in enumerate(sampling_methods):
        variances_list = [variances[method][metric] for metric in metrics]
        plt.bar(r + i * bar_width, variances_list, width=bar_width, label=method.capitalize())

    # Adding the aesthetics
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Variance', fontweight='bold')
    plt.title('Variance of Metrics across Sampling Methods')
    plt.xticks([r + bar_width for r in range(len(metrics))], metrics)
    plt.legend()
    
    plt.show()
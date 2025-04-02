import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta
import os

# TRY THIS EXPERIMENT WITHOUT RL.

# Function to compute 95% confidence intervals using a Beta distribution
def proportion_ci(n, p):
    a = n * p + 0.5
    b = n * (1 - p) + 0.5
    p0_025 = beta.isf(1 - 0.025, a, b)  # 2.5% percentile
    p0_975 = beta.isf(1 - 0.975, a, b)  # 97.5% percentile
    return pd.Series({'p0_025': p0_025, 'p0_975': p0_975})

# Labels for different n-shot settings
n_shot_levels = ["0-shot", "1-shot", "5-shot", "10-shot", "15-shot", "20-shot", "25-shot"]  

n_trials = 26  
ser_results = np.array([7.69, 3.85, 3.85, 11.54, 11.54, 11.54, 15.38]) / 100  # Convert % to proportion

# Compute confidence intervals
ci = [proportion_ci(n_trials, p) for p in ser_results]
ci = pd.DataFrame(ci)

# Convert to percentages
ser_results *= 100
ci *= 100
errors = np.abs(ci[['p0_025', 'p0_975']].sub(ser_results, axis=0).T.values)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define bar width and positions
x = np.arange(len(n_shot_levels))
bar_width = 0.5  

# Plot bars with error bars
ax.bar(x, ser_results, width=bar_width, color='#4C72B0', edgecolor='black', 
       alpha=0.85, label="EX Performance", yerr=errors, capsize=6, error_kw={'elinewidth': 2})

# Add grid for readability
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(n_shot_levels, fontsize=14)
ax.set_yticks(np.arange(0, 40, 10))
ax.set_ylim(0, 40)
ax.set_yticklabels(np.arange(0, 40, 10), fontsize=15)

# Labels and title
ax.set_ylabel("EX (%)", fontsize=15, labelpad=10)
ax.set_xlabel("n-Shot Learning", fontsize=15, labelpad=10)
ax.set_title("Effect of Increasing n-Shot on EX Performance", fontsize=16, pad=15)

save_path = "Visualise/nshot/n_shot_ex_performance.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Show and save the plot
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

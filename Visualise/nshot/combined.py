import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta
import os

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

# NER Performance Data
ner_results = np.array([53.85, 57.69, 80.77, 65.38, 76.92, 84.62, 96]) / 100  # Convert % to proportion
ci_ner = pd.DataFrame([proportion_ci(n_trials, p) for p in ner_results]) * 100
ner_results *= 100
errors_ner = np.abs(ci_ner[['p0_025', 'p0_975']].sub(ner_results, axis=0).T.values)

# PEX Performance Data
pex_results = np.array([11.54, 15.38, 23.08, 23.08, 23.08, 30.77, 30.77]) / 100  # Convert % to proportion
ci_pex = pd.DataFrame([proportion_ci(n_trials, p) for p in pex_results]) * 100
pex_results *= 100
errors_pex = np.abs(ci_pex[['p0_025', 'p0_975']].sub(pex_results, axis=0).T.values)

# Set up figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Common X positions for bars
x = np.arange(len(n_shot_levels))
bar_width = 0.5  

# Plot NER Performance
axes[0].bar(x, ner_results, width=bar_width, color='#4C72B0', edgecolor='black', alpha=0.85,
            label="NER Performance", yerr=errors_ner, capsize=6, error_kw={'elinewidth': 2})
axes[0].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
axes[0].set_ylabel("NER (%)", fontsize=15, labelpad=10)
axes[0].set_title("Effect of Increasing n-Shot on NER Performance", fontsize=16, pad=10)
axes[0].set_yticks(np.arange(0, 110, 10))
axes[0].set_ylim(0, 105)
axes[0].tick_params(axis='y', labelsize=15)  # Increase font size for y-tick labels

# Plot PEX Performance
axes[1].bar(x, pex_results, width=bar_width, color='#D62728', edgecolor='black', alpha=0.85,
            label="PEX Performance", yerr=errors_pex, capsize=6, error_kw={'elinewidth': 2})
axes[1].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
axes[1].set_ylabel("PEX (%)", fontsize=15, labelpad=10)
axes[1].set_title("Effect of Increasing n-Shot on PEX Performance", fontsize=16, pad=10)
axes[1].set_yticks(np.arange(0, 60, 5))
axes[1].set_ylim(0, 60)
axes[1].tick_params(axis='y', labelsize=15)  # Increase font size for y-tick labels

# Common x-axis formatting
axes[1].set_xticks(x)
axes[1].set_xticklabels(n_shot_levels, fontsize=14)
axes[1].set_xlabel("n-Shot Learning", fontsize=15, labelpad=10)

# Adjust layout and save the figure
plt.tight_layout()
save_path = "Visualise/nshot/n_shot_combined_performance.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()

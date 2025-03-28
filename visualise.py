import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta

# Function to compute 95% confidence intervals using a Beta distribution
def proportion_ci(n, p):
    a = n * p + 0.5
    b = n * (1 - p) + 0.5
    p0_025 = beta.isf(1 - 0.025, a, b)  # 2.5% percentile
    p0_975 = beta.isf(1 - 0.975, a, b)  # 97.5% percentile
    return pd.Series({'p0_025': p0_025, 'p0_975': p0_975})

# Labels for different conditions
conditions = ["Rules", "MSL", "Schema", 
              "Schema + Rules \n+ MSL", "Schema + Rules \n+ MSL + RL"]

# Experiment results (converted to proportions)
n_trials = 26  
results_50 = np.array([0, 7, 10, 18, 25]) / n_trials  

# Compute confidence intervals
ci_50 = [proportion_ci(n_trials, p) for p in results_50]
ci_50 = pd.DataFrame(ci_50)

# Convert to percentages
results_50 *= 100
ci_50 *= 100
errors = np.abs(ci_50[['p0_025', 'p0_975']].sub(results_50, axis=0).T.values)

# Set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define bar width and positions
x = np.arange(len(conditions))
bar_width = 0.5  

# Plot bars with error bars
ax.bar(x, results_50, width=bar_width, color='#4C72B0', edgecolor='black', 
       alpha=0.85, label="Non-Empty Output Rate", yerr=errors, capsize=6, error_kw={'elinewidth': 2})

# Add grid for readability
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=14)
ax.set_yticks(np.arange(0, 110, 10))
ax.set_ylim(0, 105)
ax.set_yticklabels(np.arange(0, 110, 10), fontsize=15)

# Labels and title
ax.set_ylabel("NER (%)", fontsize=15, labelpad=10)
ax.set_xlabel("Condition", fontsize=15, labelpad=10)
# ax.set_title("Effect of Different Conditions on NER", fontsize=16, pad=15)

# Show and save the plot
plt.tight_layout()
plt.savefig("nor.png", dpi=300)
plt.show()
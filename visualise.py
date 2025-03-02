import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta

# Function to compute 95% confidence intervals
def proportion_ci(n, p):
    a = n * p + 0.5
    b = n * (1 - p) + 0.5
    p0_025 = beta.isf(1 - 0.025, a, b)  # 2.5% centile
    p0_975 = beta.isf(1 - 0.975, a, b)  # 97.5% centile
    return pd.Series({'p0_025': p0_025, 'p0_975': p0_975})

# Labels for different conditions
conditions = ["Only Rules", "Only Training", "Only Schema", 
              "(Schema + Rules + Training)", "(Schema + Rules + Training + RL)"]

# Convert to percentages
n_trials = 26  # Number of trials
results_50 = np.array([0, 7, 10, 18, 25]) / n_trials  # Proportion

# Compute confidence intervals
ci_50 = [proportion_ci(n_trials, p) for p in results_50]
ci_50 = pd.DataFrame(ci_50)

# Convert to percentage values
results_50 *= 100
ci_50 *= 100
errors = np.abs(ci_50[['p0_025', 'p0_975']].sub(results_50, axis=0).T.values)

# Set up x-axis positions
x = np.arange(len(conditions))
bar_width = 0.4  

# Create figure and axis object
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars with error bars
bars = ax.bar(x, results_50, width=bar_width, label="50% Training Set", 
              color='skyblue', zorder=3, yerr=errors, capsize=5, error_kw={'elinewidth': 2, 'markeredgewidth': 2})

# Add grey grid
ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=0, ha="center")
ax.set_ylabel("Non-Empty Output Rate (%)")
ax.set_title("Non-Empty Output Rate with Different Conditions")
ax.legend()

plt.tight_layout()
plt.savefig("nor.png", dpi=300)
plt.show()

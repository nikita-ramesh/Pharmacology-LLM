import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import os

# Function to compute 95% confidence intervals using a Beta distribution
def proportion_ci(n, p):
    a = n * p + 0.5
    b = n * (1 - p) + 0.5
    p0_025 = beta.isf(1 - 0.025, a, b)  # 2.5% percentile
    p0_975 = beta.isf(1 - 0.975, a, b)  # 97.5% percentile
    return p0_025 * 100, p0_975 * 100

# Define metrics and their performance values
metrics = ["NER", "SER", "PEX", "EX"]
n_trials = 30  # Number of trials

# Performance values (in percentages)
without_schema = np.array([61.54, 92.31, 3.85, 3.85])  # Replace with real data
with_schema = np.array([96.15, 100.00, 30.77, 15.38])  

# Compute confidence intervals
ci_without = np.array([proportion_ci(n_trials, p / 100) for p in without_schema])
ci_with = np.array([proportion_ci(n_trials, p / 100) for p in with_schema])

# Convert confidence intervals to error margins
errors_without = np.abs(ci_without - without_schema[:, None]).T
errors_with = np.abs(ci_with - with_schema[:, None]).T

# Create DataFrame for Seaborn
df = pd.DataFrame({
    "Metric": np.tile(metrics, 2),
    "Performance": np.concatenate([without_schema, with_schema]),
    "Many-shot Learning": ["Without"] * 4 + ["With"] * 4,
    "Lower CI": np.concatenate([ci_without[:, 0], ci_with[:, 0]]),
    "Upper CI": np.concatenate([ci_without[:, 1], ci_with[:, 1]])
})

# Set Seaborn style for beauty ✨
sns.set(style="whitegrid", palette="pastel")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Barplot with error bars
sns.barplot(
    data=df, x="Metric", y="Performance", hue="Many-shot Learning", capsize=0.15, errwidth=2, 
    ci=None, dodge=True, edgecolor="black", ax=ax, palette=["#D62728", "#4C72B0"]
)

# Add error bars manually (since Seaborn does not support CI input directly)
for i in range(len(metrics)):
    ax.errorbar(i - 0.2, without_schema[i], yerr=[[errors_without[0][i]], [errors_without[1][i]]], fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.errorbar(i + 0.2, with_schema[i], yerr=[[errors_with[0][i]], [errors_with[1][i]]], fmt='none', color='black', capsize=5, linewidth=1.5)

# Labels and title
ax.set_ylabel("Performance (%)", fontsize=14)
ax.set_xlabel("")
ax.set_title("Impact of Many-shot Learning on Different Metrics", fontsize=16, pad=10)
ax.set_ylim(0, 100)  
ax.legend(fontsize=12, title="Many-shot Learning")

# Beautify the grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.set_xticklabels(metrics, fontsize=14)

save_path = "Visualise/significance/results/msl_importance.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Show and save the plot
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

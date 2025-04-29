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

# Define metrics and trials
metrics = ["NER", "SER", "PEX", "EX"]
n_trials = 30  # Number of trials

# Data for each experiment
experiments = {
    "Many-shot Learning": {
        "without": np.array([61.54, 92.31, 3.85, 3.85]),
        "with": np.array([96.15, 100.00, 30.77, 15.38])
    },
    "Self-Correction": {
        "without": np.array([65.38, 96.15, 23.08, 11.54]),
        "with": np.array([96.15, 100.00, 30.77, 15.38])
    },
    "Rules": {
        "without": np.array([73.08, 96.15, 34.62, 7.69]),
        "with": np.array([96.15, 100.00, 30.77, 15.38])
    },
    "Schema": {
        "without": np.array([69.23, 88.46, 26.92, 15.38]),
        "with": np.array([96.15, 100.00, 30.77, 15.38])
    }
}

# Set up figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
sns.set(style="whitegrid", palette="pastel")

# Flatten axes for easy iteration
axes = axes.flatten()

# Loop through experiments and plot each in its respective subplot
for ax, (title, data) in zip(axes, experiments.items()):
    without_schema = data["without"]
    with_schema = data["with"]
    
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
        title: ["Without"] * 4 + ["With"] * 4,
        "Lower CI": np.concatenate([ci_without[:, 0], ci_with[:, 0]]),
        "Upper CI": np.concatenate([ci_without[:, 1], ci_with[:, 1]])
    })

    # Barplot with error bars
    sns.barplot(
        data=df, x="Metric", y="Performance", hue=title, capsize=0.15, errwidth=2, 
        ci=None, dodge=True, edgecolor="black", ax=ax, palette=["#D62728", "#4C72B0"]
    )

    # Add error bars manually
    for i in range(len(metrics)):
        ax.errorbar(i - 0.2, without_schema[i], yerr=[[errors_without[0][i]], [errors_without[1][i]]], fmt='none', color='black', capsize=5, linewidth=1.5)
        ax.errorbar(i + 0.2, with_schema[i], yerr=[[errors_with[0][i]], [errors_with[1][i]]], fmt='none', color='black', capsize=5, linewidth=1.5)

    # Labels and title
    ax.set_title(f"Ablation of {title}", fontsize=14, pad=7)  # Increased padding
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Performance", fontsize=12)
    ax.legend(fontsize=11, title=title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels(metrics, fontsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12)

# Adjust layout to prevent overlap
plt.subplots_adjust(hspace=0.15, wspace=0.06)

# Save and show
save_path = "Visualise/significance/results/combined_importance.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

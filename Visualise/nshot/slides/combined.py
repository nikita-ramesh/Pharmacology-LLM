import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta
import os

# Function to compute 95% confidence intervals using a Beta distribution
def proportion_ci(n, p):
    a = n * p + 0.5
    b = n * (1 - p) + 0.5
    p0_025 = beta.isf(1 - 0.025, a, b)
    p0_975 = beta.isf(1 - 0.975, a, b)
    return pd.Series({'p0_025': p0_025, 'p0_975': p0_975})

# Labels for different n-shot settings
n_shot_levels = ["0-shot", "1-shot", "5-shot", "10-shot", "15-shot", "20-shot", "25-shot"]
n_trials = 26

# Function to calculate results, confidence intervals, and error bars
def prepare_data(results):
    results = np.array(results) / 100
    ci = pd.DataFrame([proportion_ci(n_trials, p) for p in results]) * 100
    results *= 100
    errors = np.abs(ci[['p0_025', 'p0_975']].sub(results, axis=0).T.values)
    return results, errors

# Prepare data
ner_results, errors_ner = prepare_data([53.85, 57.69, 80.77, 65.38, 76.92, 84.62, 96])
pex_results, errors_pex = prepare_data([11.54, 15.38, 23.08, 23.08, 23.08, 30.77, 30.77])
ex_results, errors_ex = prepare_data([7.69, 3.85, 3.85, 11.54, 11.54, 11.54, 15.38])

# Bar chart plotting function
def plot_and_save(title, ylabel, data, errors, colour, yticks, ylim, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(n_shot_levels))
    bar_width = 0.5

    ax.bar(x, data, width=bar_width, color=colour, edgecolor='black', alpha=0.85,
           yerr=errors, capsize=6, error_kw={'elinewidth': 2})
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(n_shot_levels, fontsize=14)
    ax.set_yticks(np.arange(*yticks))
    ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel("n-Shot Learning", fontsize=15, labelpad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

# Save plots
plot_and_save(
    title="Effect of Increasing n-Shot on NER Performance",
    ylabel="NER (%)",
    data=ner_results,
    errors=errors_ner,
    colour="#4C72B0",
    yticks=(0, 110, 10),
    ylim=(0, 105),
    filename="Visualise/nshot/slides/n_shot_ner.png"
)

plot_and_save(
    title="Effect of Increasing n-Shot on PEX Performance",
    ylabel="PEX (%)",
    data=pex_results,
    errors=errors_pex,
    colour="#D62728",
    yticks=(0, 60, 5),
    ylim=(0, 60),
    filename="Visualise/nshot/slides/n_shot_pex.png"
)

plot_and_save(
    title="Effect of Increasing n-Shot on EX Performance",
    ylabel="EX (%)",
    data=ex_results,
    errors=errors_ex,
    colour="#9467BD",
    yticks=(0, 50, 10),
    ylim=(0, 40),
    filename="Visualise/nshot/slides/n_shot_ex.png"
)

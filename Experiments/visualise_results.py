import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert to percentages
results_50 = np.array([0, 7, 10, 10, 10, 15, 15, 18]) * 100 / 26
results_75 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10]) * 100 / 13

# Binary indicators for Schema, Rules, and Training
schema = np.array([0, 0, 1, 1, 0, 1, 1, 1])  # 1 = Schema Present, 0 = No Schema
rules = np.array([0, 1, 0, 1, 1, 1, 0, 1])  # 1 = Rules Present, 0 = No Rules
training = np.array([0, 1, 0, 0, 1, 1, 1, 1])  # 1 = Training Present, 0 = No Training

# Create DataFrame
df = pd.DataFrame({
    "Schema": schema,
    "Rules": rules,
    "Training Set": training,
    "50% Training Set": results_50,
    "75% Training Set": results_75
})

# Pivot table without duplicate index errors
heatmap_data_50 = df.pivot(index=["Schema", "Rules"], columns="Training Set", values="50% Training Set")
heatmap_data_75 = df.pivot(index=["Schema", "Rules"], columns="Training Set", values="75% Training Set")

# Rename training columns
heatmap_data_50.columns = ["No Training Set", "Training Set"]
heatmap_data_75.columns = ["No Training Set", "Training Set"]

# Create subplots for side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for 50% Training Set
sns.heatmap(heatmap_data_50, annot=True, cmap="Blues", fmt=".1f", linewidths=0.5, cbar=True, ax=axes[0])
axes[0].set_title("50% Training Set Performance")
axes[0].set_xlabel("")
axes[0].set_ylabel("Schema & Rules")

# Heatmap for 75% Training Set
sns.heatmap(heatmap_data_75, annot=True, cmap="Oranges", fmt=".1f", linewidths=0.5, cbar=True, ax=axes[1])
axes[1].set_title("75% Training Set Performance")
axes[1].set_xlabel("")
axes[1].set_ylabel("Schema & Rules")

# Adjust layout and show plot
plt.tight_layout()
plt.show()

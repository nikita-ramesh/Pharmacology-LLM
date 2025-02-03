import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Labels for different conditions
conditions = [
    "No schema, No training", "No schema, No rules, Yes training", 
    "Yes schema, No rules, No training", "Yes schema, Yes rules, No training",
    "No schema, Yes rules, Yes training", "Yes schema, Yes rules (at bottom), Yes training",
    "Yes schema, No rules, Yes training", "Yes schema, Yes rules, Yes training"
]

# Convert to percentages
results_50 = np.array([0, 7, 10, 10, 10, 15, 15, 18]) * 100 / 26
results_75 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10]) * 100 / 13

# Define structured factors
schema = np.array([0, 0, 1, 1, 0, 1, 1, 1])  # 1 = Schema Present, 0 = No Schema
rules = np.array([0, 0, 0, 1, 1, 1, 0, 1])  # 1 = Rules Present, 0 = No Rules
training = np.array([0, 1, 0, 0, 1, 1, 1, 1])  # 1 = Training Present, 0 = No Training

# Create DataFrame
df = pd.DataFrame({
    "Schema": schema,
    "Rules": rules,
    "Training": training,
    "50% Training Set": results_50,
    "75% Training Set": results_75
})

# Pivot tables separately for heatmaps
heatmap_data_50 = df.pivot_table(values="50% Training Set", index=["Schema", "Rules"], columns="Training")
heatmap_data_75 = df.pivot_table(values="75% Training Set", index=["Schema", "Rules"], columns="Training")

# Create subplots for side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for 50% Training Set
sns.heatmap(heatmap_data_50, annot=True, cmap="Blues", fmt=".1f", linewidths=0.5, cbar=True, ax=axes[0])
axes[0].set_title("50% Training Set Performance")
axes[0].set_xlabel("Training Presence (0 = No, 1 = Yes)")
axes[0].set_ylabel("Schema & Rules (Index = Schema, Rules)")

# Heatmap for 75% Training Set
sns.heatmap(heatmap_data_75, annot=True, cmap="Oranges", fmt=".1f", linewidths=0.5, cbar=True, ax=axes[1])
axes[1].set_title("75% Training Set Performance")
axes[1].set_xlabel("Training Presence (0 = No, 1 = Yes)")
axes[1].set_ylabel("Schema & Rules (Index = Schema, Rules)")

# Adjust layout and show plot
plt.tight_layout()
plt.show()

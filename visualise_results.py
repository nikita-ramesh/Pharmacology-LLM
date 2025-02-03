import numpy as np
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

# Set up x-axis positions
x = np.arange(len(conditions))

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot 50% training results with a line and dots
ax.plot(x, results_50, marker='o', linestyle='-', color='skyblue', label="50% Training Set", zorder=3)

# Plot 75% training results with a different marker
ax.scatter(x, results_75, color='orange', label="75% Training Set", zorder=3, s=80, edgecolors='black')

# Add horizontal gridlines for readability
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=45, ha="right")
ax.set_ylabel("Correct SQL Queries (%)")
ax.set_title("SQL Query Performance with Different Conditions")
ax.legend()
plt.tight_layout()

# Show plot
plt.show()

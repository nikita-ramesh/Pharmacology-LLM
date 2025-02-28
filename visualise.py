import numpy as np
import matplotlib.pyplot as plt

# Labels for different conditions
conditions = [
    # "No schema, No training", "No schema, No rules, Yes training", 
    # "Yes schema, No rules, No training", "Yes schema, Yes rules, No training",
    # "No schema, Yes rules, Yes training", "Yes schema, Yes rules (at bottom), Yes training",
    # "Yes schema, No rules, Yes training", "Yes schema, Yes rules, Yes training"

    "Only Rules", "Only Training", "Only Schema", 
    "(Schema + Rules + Training)", "(Schema + Rules + Training + RL)"
]

# Convert to percentages
results_50 = np.array([0, 7, 10, 18, 25]) * 100 / 26
# results_75 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan]) * 100 / 13
# results_75 = np.array([np.nan, np.nan, np.nan, np.nan, 10]) * 100 / 13

# Set up x-axis positions
x = np.arange(len(conditions))
bar_width = 0.4  # Adjust bar width

# Create figure and axis object
fig, ax = plt.subplots(figsize=(12, 6))

# Plot all 50% bars centered
bars = ax.bar(x, results_50, width=bar_width, label="50% Training Set", color='skyblue', zorder=3)

# Plot 75% Training Set only for the last condition, side by side
# ax.bar(x[-1] - bar_width/2, results_50[-1], width=bar_width, color='skyblue', zorder=3)  # Replot last 50% bar slightly left
# ax.bar(x[-1] + bar_width/2, results_75[-1], width=bar_width, label="75% Training Set", color='orange', zorder=3)  # 75% bar right

# Add subtle grey grid behind the bars (zorder=0 ensures it's behind the bars)
ax.grid(axis='y', color='grey', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=0, ha="center")
ax.set_ylabel("Non-Empty Output Rate (%)")
ax.set_title("Non-Empty Output Rate with Different Conditions")
ax.legend()

# Add percentage labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}%", ha='center', va='bottom')

plt.tight_layout()

# Save the plot
plt.savefig("nor.png", dpi=300)

# Show plot
plt.show()

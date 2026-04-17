import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

seed = sys.argv[1]


def smooth(vals, window):
    """Smooths values using a sliding window."""

    if window > 1:
        if window > len(vals):
            window = len(vals)
        y = np.ones(window)
        x = vals
        z = np.ones(len(vals))
        mode = "same"
        vals = np.convolve(x, y, mode) / np.convolve(z, y, mode)

    return vals


df = pd.read_csv(f"{seed}/log.csv")
steps = df["train/steps"].to_numpy()

# Extract all rwd_metric mean columns
rwd_metrics = [
    "forward_direction",
    "forward_lean",
    "y_vel",
    "gaussian_vel",
    "gaussian_x_vel",
    "gaussian_plateau_y_vel",
    "sideways_lean",
]
cost_metrics = [
    "done",
    "grf",
    "joint_limit",
    "number_muscles",
    "smooth_exc",
    "self_contact",
    "x_drift",
]
test_rwd_cols = [f"test/rwd_metrics/{m}/mean" for m in rwd_metrics]
test_cost_cols = [f"test/rwd_metrics/{m}/mean" for m in cost_metrics]

rwd_labels = [
    c.replace("test/rwd_metrics/", "").replace("/mean", "")
    for c in test_rwd_cols
    if c in df
]
cost_labels = [
    c.replace("test/rwd_metrics/", "").replace("/mean", "")
    for c in test_cost_cols
    if c in df
]

# Separate positive and negative contributions
data_pos = (
    np.array([df[c].to_numpy() for c in test_rwd_cols if c in df])
    / df["test/episode_length/mean"].to_numpy()
)
data_neg = (
    np.array([df[c].to_numpy() for c in test_cost_cols if c in df])
    / df["test/episode_length/mean"].to_numpy()
)

window = 5
data_pos = [smooth(d, window) for d in data_pos]
data_neg = [smooth(d, window) for d in data_neg]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Normalised Reward Metrics", fontsize=14)

colors = plt.cm.tab20(np.linspace(0, 1, len(rwd_labels) + len(cost_labels)))
# Stacked positive
ax1.stackplot(
    steps,
    data_pos,
    labels=rwd_labels,
    colors=colors[: len(rwd_labels)],
    alpha=0.8,
)
ax1.set_ylabel("Positive Contributions")
ax1.set_title("Reward Components (Positive)")
ax1.legend(loc="upper left", fontsize=7, ncol=2)
ax1ymin, ax1ymax = ax1.get_ylim()

# Stacked negative
ax2.stackplot(
    steps,
    data_neg,
    labels=cost_labels,
    colors=colors[len(rwd_labels) :],
    alpha=0.8,
)
ax2.set_ylabel("Negative Contributions")
ax2.set_title("Reward Components (Negative)")
ax2.set_xlabel("Steps")
ax2.legend(loc="lower left", fontsize=7, ncol=2)
ax2ymin, ax2ymax = ax2.get_ylim()

ax1.set_ylim(ymin=0, ymax=max(10, max(abs(ax1ymax), abs(ax2ymin))))
ax2.set_ylim(ymin=-max(10, max(abs(ax1ymax), abs(ax2ymin))), ymax=0)

plt.tight_layout()
plt.savefig(f"{seed}/plots/rewards-stacked-normed.png", dpi=150)

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

fig, axes = plt.subplots(4, 3, figsize=(16, 16))
fig.suptitle("Reward Metrics", fontsize=14)
steps = df["train/steps"].to_numpy()
# Extract all rwd_metric mean columns
rwd_metrics = [
    "forward_direction",
    "forward_lean",
    "gaussian_vel",
    "gaussian_vel_x",
    "gaussian_vel_y",
    "sideways_lean",
    "done",
    "grf",
    "joint_limit",
    "number_muscles",
    "smooth_exc",
]

rwd_data = [
    (
        metric,
        np.array(
            smooth(
                df[col].to_numpy() / df["test/episode_length/mean"].to_numpy(),
                5,
            )
        ),
    )
    for metric, col in [(m, f"test/rwd_metrics/{m}/mean") for m in rwd_metrics]
    if col in df
]

for idx, (lbl, data) in enumerate(rwd_data):
    ax = axes[idx // 3, idx % 3]
    ax.plot(steps, data, label=lbl)
    ax.set_title(lbl)
    ax.set_xlabel("Steps")
    ax.legend()
    # ax.set_ylim(ymin=-5, ymax=5)
    y0, _ = ax.get_ylim()
    if y0 >= 0:
        ax.set_ylim(ymin=0)
    else:
        ax.set_ylim(ymax=0)


plt.tight_layout()
plt.savefig(f"{seed}/plots/rewards.png", dpi=150)

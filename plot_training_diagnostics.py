import sys

import matplotlib.pyplot as plt
import pandas as pd

seed = sys.argv[1]

df = pd.read_csv(f"{seed}/log.csv")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("MPO Training Diagnostics", fontsize=14)
steps = df["train/steps"].to_numpy()

# 1. Episode length train vs test
ax = axes[0, 0]
ax.plot(steps, df["train/episode_length/mean"].to_numpy(), label="train")
ax.fill_between(
    steps,
    df["train/episode_length/mean"] - df["train/episode_length/std"],
    df["train/episode_length/mean"] + df["train/episode_length/std"],
    alpha=0.2,
)
ax.plot(steps, df["test/episode_length/mean"].to_numpy(), label="test")
ax.fill_between(
    steps,
    df["test/episode_length/mean"] - df["test/episode_length/std"],
    df["test/episode_length/mean"] + df["test/episode_length/std"],
    alpha=0.2,
)
ax.set_title("Episode Length (Train vs Test)")
ax.set_xlabel("Steps")
ax.legend()

# 2. Episode score train vs test
ax = axes[0, 1]
ax.plot(steps, df["train/episode_score/mean"].to_numpy(), label="train")
ax.plot(steps, df["test/episode_score/mean"].to_numpy(), label="test")
ax.set_title("Episode Score (Train vs Test)")
ax.set_xlabel("Steps")
ax.legend()

# 3. KL losses — key MPO diagnostic
ax = axes[0, 2]
ax.plot(steps, df["actor/kl_mean_loss"].to_numpy(), label="kl_mean")
ax.plot(steps, df["actor/kl_std_loss"].to_numpy(), label="kl_std")
ax.set_title("KL Losses")
ax.set_xlabel("Steps")
ax.legend()

# 4. Alpha (dual variables for KL constraints)
ax = axes[1, 0]
ax.plot(steps, df["actor/alpha_mean"].to_numpy(), label="alpha_mean")
ax.plot(steps, df["actor/alpha_std"].to_numpy(), label="alpha_std")
ax.set_title("Alpha (KL Dual Variables)")
ax.set_xlabel("Steps")
ax.legend()

# 5. Temperature — controls action sharpness/entropy
ax = axes[1, 1]
ax.plot(steps, df["actor/temperature"].to_numpy(), label="temperature")
ax.plot(
    steps,
    df["actor/penalty_temperature"].to_numpy(),
    label="penalty_temp",
    linestyle="--",
)
ax.set_title("Temperature (proxy for entropy)")
ax.set_xlabel("Steps")
ax.legend()

# 6. Action std (train) — direct entropy proxy
ax = axes[1, 2]
ax.plot(steps, df["train/action/std"].to_numpy(), label="train action std")
ax.plot(steps, df["test/action/std"].to_numpy(), label="test action std")
ax.set_title("Action Std (Train vs Test)")
ax.set_xlabel("Steps")
ax.legend()

# 7. Termination rate
ax = axes[2, 0]
ax.plot(steps, df["test/terminated/mean"].to_numpy())
ax.set_title("Termination Rate (Test)")
ax.set_xlabel("Steps")


# 8. Key reward components
ax = axes[2, 1]
for rwd_stat in [
    "gaussian_vel",
    "gaussian_x_vel",
    "gaussian_plateau_y_vel",
    "grf",
    "smooth_exc",
    "number_muscles",
    "joint_limit",
    "self_contact",
    "x_drift",
]:
    col = f"test/rwd_metrics/{rwd_stat}/mean"
    if col in df:
        ax.plot(
            steps,
            df[col].to_numpy(),
            label=rwd_stat,
        )
ax.set_title("Reward Components (Test)")
ax.set_xlabel("Steps")
ax.legend()

# 9. Critic loss
ax = axes[2, 2]
ax.plot(steps, df["critic/loss"].to_numpy())
ax.set_title("Critic Loss")
ax.set_xlabel("Steps")

plt.tight_layout()
plt.savefig(f"{seed}/plots/training-diagnostics.png", dpi=150)

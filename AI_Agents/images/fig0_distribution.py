import numpy as np
import matplotlib.pyplot as plt

words  = ["photon", "redshift", "galaxy", "spectrum", "pulsar", "nebula", "quasar", "black hole"]
logits = np.array([3.2, 2.8, 2.1, 1.5, 1.0, 0.4, 0.1, -0.3])

T = 1.0
scaled = logits / T
exp_s  = np.exp(scaled - scaled.max())
probs  = exp_s / exp_s.sum()

fig, ax = plt.subplots(figsize=(7, 4))
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.22)

x = np.arange(len(words))
ax.bar(x, probs, color="mediumseagreen", alpha=0.85, width=0.6, edgecolor="white", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(words, rotation=40, ha="right", fontsize=10)
ax.set_ylabel("Probability", fontsize=11)
ax.set_ylim(0, 0.45)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Next-token probability distribution", fontsize=13, fontweight="bold", pad=10)

plt.savefig("fig0_distribution.png", dpi=150, bbox_inches="tight")
print("Saved fig0_distribution.png")

import numpy as np
import matplotlib.pyplot as plt

words  = ["photon", "redshift", "galaxy", "spectrum", "pulsar", "nebula", "quasar", "black hole"]
logits = np.array([3.2, 2.8, 2.1, 1.5, 1.0, 0.4, 0.1, -0.3])

temperatures = [0.3, 1.0, 2.0]
labels       = ["T = 0.3  (sharp)", "T = 1.0  (neutral)", "T = 2.0  (flat)"]
colors       = ["steelblue", "mediumseagreen", "firebrick"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97, top=0.82, bottom=0.22)

x = np.arange(len(words))

for ax, T, label, color in zip(axes, temperatures, labels, colors):
    scaled = logits / T
    exp_s  = np.exp(scaled - scaled.max())
    probs  = exp_s / exp_s.sum()

    ax.bar(x, probs, color=color, alpha=0.85, width=0.6, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=9.5)
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=8)
    ax.set_ylim(0, 0.72)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

axes[0].set_ylabel("Probability", fontsize=10.5)
fig.suptitle("Temperature and the Token Distribution", fontsize=13.5, fontweight="bold", y=0.97)

plt.savefig("fig1_temperature.png", dpi=150, bbox_inches="tight")
print("Saved fig1_temperature.png")

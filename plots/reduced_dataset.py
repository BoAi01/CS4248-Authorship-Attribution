import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()

ax.plot([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], 
    [0.5607638889,
    0.5573630137,
    0.5505681818,
    0.5881849315,
    0.5887978142,
    0.6028409091,
    0.5964355469,
    0.6028156997],
label="BERT")

ax.plot([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    [0.6527777778,
    0.6489726027,
    0.6545454545,
    0.6601027397,
    0.6646174863,
    0.6590909091,
    0.6555175781,
    0.6604095563,],
label="BERT-Contrastive")

ax.set_xticks(np.arange(0.125, 1.01, 0.125))
ax.set_yticks(
    np.arange(0.54, 0.67, 0.02), 
    )
ax.set_yticklabels(labels=["54.0", "56.0", "58.0", "60.0", "62.0", "64.0", "66.0"])
ax.set_xlabel("Fraction of dataset size")
ax.set_ylabel("Accuracy (%)")
ax.legend()

plt.show()

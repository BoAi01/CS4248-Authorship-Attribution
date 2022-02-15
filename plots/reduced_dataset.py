import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()

ax.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.7680151707,
    0.7888655462,
    0.7962857143,
    0.7976547441,
    ],
label="TuringBench Contra-X",
color='indianred'
)

ax.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.7583227981,
    0.7772058824,
    0.7904061625,
    0.7921664168
    ],
label="TuringBench base",
color='indianred',
linestyle='dashed')

ax.plot(
    #[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    [0.25, 0.5, 0.75, 1],
    [
    #0.6527777778,
    0.6489726027,
    #0.6545454545,
    0.6601027397,
    #0.6646174863,
    0.6590909091,
    #0.6555175781,
    0.6604095563,
    ],
label="BLOG10 Contra-X",
color='steelblue'
)

ax.plot(
    #[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    [0.25, 0.5, 0.75, 1],
    [
    #0.5607638889,
    0.5573630137,
    #0.5505681818,
    0.5881849315,
    #0.5887978142,
    0.6028409091,
    #0.5964355469,
    0.6028156997
    ],
label="BLOG10 base",
color='steelblue',
linestyle='dashed')

ax.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.5282079646,
    0.5756578947,
    0.608369883,
    0.6199508734,
    ],
label="BLOG50 Contra-X",
color='olivedrab'
)

ax.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.517699115,
    0.5301535088,
    0.5561038012,
    0.5518558952
    ],
label="BLOG50 base",
color='olivedrab',
linestyle='dashed')

ax.set_xticks(np.arange(0.25, 1.01, 0.25))
ax.set_yticks(
    np.arange(0.50, 0.80, 0.05), 
    )
ax.set_yticklabels(labels=["50.0", "55.0", "60.0", "65.0", "70.0", "75.0", "80.0"])
ax.set_xlabel("Fraction of dataset size")
ax.set_ylabel("Accuracy (%)")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))

plt.show()

plt.savefig("shrinked_plot.png", dpi=200)


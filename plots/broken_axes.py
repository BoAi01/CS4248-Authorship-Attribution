import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np
fig = plt.figure(figsize=(6,4))
baxes = brokenaxes(xlims=((0,0.025),(0.22,1.0)), ylims=((0,0.015),(0.48,0.8)), hspace=.06, wspace=.04)
x = np.linspace(0.25,1,4)
baxes.plot(
[0.25, 0.5, 0.75, 1],
    [
    0.7680151707,
    0.7888655462,
    0.7962857143,
    0.7976547441,
    ],
#label="TuringBench Contra-X",
color='indianred'
)

baxes.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.7583227981,
    0.7772058824,
    0.7904061625,
    0.7921664168
    ],
#label="TuringBench base",
color='indianred',
linestyle='dashed')

baxes.plot(
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
#label="Blog10 Contra-X",
color='steelblue'
)

baxes.plot(
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
#label="Blog10 base",
color='steelblue',
linestyle='dashed')

baxes.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.5282079646,
    0.5756578947,
    0.608369883,
    0.6199508734,
    ],
#label="Blog50 Contra-X",
color='olivedrab'
)

baxes.plot(
    [0.25, 0.5, 0.75, 1],
    [
    0.517699115,
    0.5301535088,
    0.5561038012,
    0.5518558952
    ],
#label="Blog50 base",
color='olivedrab',
linestyle='dashed')

# baxes.legend(loc="best")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='black', label='Contra-BERT'),
                   Line2D([0], [0], color='black', label='BERT', linestyle='dashed'),
                   Line2D([0], [0], marker='o', color='indianred', label='TuringBench',
                          markerfacecolor='indianred', markersize=8, linestyle = 'None'),
                   Line2D([0], [0], marker='o', color='steelblue', label='Blog10',
                          markerfacecolor='steelblue', markersize=8, linestyle = 'None'),
                   Line2D([0], [0], marker='o', color='olivedrab', label='Blog50',
                          markerfacecolor='olivedrab', markersize=8, linestyle = 'None')]
baxes.axs[1].legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 0.97))
baxes.axs[1].set_xticks(np.arange(0.25, 1.01, 0.25))
baxes.axs[3].set_xticks(np.arange(0.25, 1.01, 0.25))
baxes.axs[0].set_yticks(
    np.arange(0.50, 0.80, 0.05), 
    )
baxes.axs[0].set_yticklabels(labels=["50.0", "55.0", "60.0", "65.0", "70.0", "75.0", "80.0"])
baxes.axs[1].set_yticks(
    np.arange(0.50, 0.80, 0.05), 
    )
baxes.axs[1].set_yticklabels(labels=["50.0", "55.0", "60.0", "65.0", "70.0", "75.0", "80.0"])
baxes.axs[2].set_yticklabels(labels=[""])
baxes.set_xlabel("Fraction of dataset size", labelpad=20)
baxes.set_ylabel("Accuracy (%)", labelpad=34)


# # Create the figure
# fig, ax = plt.subplots()

plt.show()
plt.savefig("shrinked_plot.png", dpi=200)




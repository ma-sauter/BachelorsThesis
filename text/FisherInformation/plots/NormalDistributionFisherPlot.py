import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathrsfs}",
        "font.family": "Helvetica",
        "font.size": "12",
    }
)


def gauss(x, mu, sigma):
    return (
        1
        / np.sqrt(2 * np.pi * sigma**2)
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


xlist = np.linspace(0, 5, 200)

fig, ax = plt.subplots(1, 2, figsize=(390 / 72, 390 / 3 / 72))

sigma_list = [0.3, 1.5]
for i in range(2):
    ylist = gauss(xlist, 2, sigma_list[i])
    ax[i].plot(xlist, ylist, c=cm.magma(0.4))

    random_val = np.random.normal(loc=2, scale=sigma_list[i], size=4)
    while np.any(random_val > 5) or np.any(
        random_val < 0
    ):  # Repull the random values so that every value is inside of the plot
        random_val = np.random.normal(loc=2, scale=sigma_list[i], size=4)

    random_val_color = "#00E88F"
    for j, x in enumerate(random_val):
        yval = gauss(x, 2, sigma_list[i])

        ax[i].plot(
            [x, x],
            [0, yval],
            "--",
            color=random_val_color,
        )
        ax[i].plot(x, yval, "x", color=random_val_color)

    ax[i].set_ylim(0, None)
    ax[i].set_xticks(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    )
    ax[i].set_xticklabels(["$0$", "$1$", "$\mu$", "$3$", "$4$", "$5$"])
    ax[i].annotate(
        text="",
        xy=(np.min(random_val), 40),
        xytext=(np.max(random_val), 40),
        xycoords=("data", "subfigure points"),
        arrowprops=dict(
            arrowstyle="<->",
            shrinkA=0,
            shrinkB=0,
            edgecolor=cm.magma(0.7),
            linewidth=1,
        ),
    )
    ax[i].annotate(
        text=f"$\sigma = {sigma_list[i]}$", xy=(0.05, 0.85), xycoords="axes fraction"
    )
ax[1].set_ylim(0, 0.31)

plt.savefig("NormalDistributionPlot.pdf", bbox_inches="tight")

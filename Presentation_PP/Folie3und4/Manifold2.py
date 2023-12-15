import matplotlib.pyplot as plt
import numpy as np

# enable latex text rendering with a serif font and font size 16
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 16})


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$p(x)$")
# adjust figure size
fig.set_size_inches(10, 5)
# add horizontal space between subplots
fig.subplots_adjust(wspace=1)


def gaussian(x, mu, sigma):
    return (
        1
        / np.sqrt(2 * np.pi * sigma**2)
        * np.exp(-1 / 2 * (x - mu) ** 2 / sigma**2)
    )


# now create a plot of mu and sigma
ax2 = fig.add_subplot(122)
ax2.set_xlim(-3, 3)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r"$\mu$")
ax2.set_ylabel(r"$\sigma$")
ax2.grid(True)

x = np.linspace(-3, 3, 200)
mu1 = 0
sigma1 = 0.9
mu2 = 0
sigma2 = 0.5
ax1.plot(x, gaussian(x, mu1, sigma1), color="red", linewidth=2)
ax1.plot(x, gaussian(x, mu2, sigma2), color="green", linewidth=2)
ax2.plot(mu1, sigma1, "o", color="red", linewidth=2)
ax2.plot(mu2, sigma2, "o", color="green", linewidth=2)
# draw an arrow between the two points in axis 2
ax2.annotate(
    "",
    xy=(mu2, sigma2),
    xytext=(mu1, sigma1),
    arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5),
)
# add text to middle of the arrow
ax2.text(
    (mu1 + mu2) / 2 + 0.3,
    (sigma1 + sigma2) / 2,
    r"$D$",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
)
# add text to axis 1
ax1.text(
    1.5,
    0.4,
    r"$D$?",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
)

plt.show()
plt.close()


# Comparison plot
fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$p(x)$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$p(x)$")
# adjust figure size
fig.set_size_inches(10, 5)
# add horizontal space between subplots
fig.subplots_adjust(wspace=1)

# now create a plot of mu and sigma
ax3.set_xlim(-3, 3)
ax3.set_ylim(0, 1)
ax3.set_xlabel(r"$\mu$")
ax3.set_ylabel(r"$\sigma$")
ax3.grid(True)

mu1 = 0
sigma1 = 0.9
mu2 = 0
sigma2 = 0.5
ax1.plot(x, gaussian(x, mu1, sigma1), color="red", linewidth=2)
ax2.plot(x, gaussian(x, mu2, sigma2), color="green", linewidth=2)
ax3.plot(mu1, sigma1, "o", color="red", linewidth=2)
ax3.plot(mu2, sigma2, "o", color="green", linewidth=2)
# draw an arrow between the two points in axis 2
"""ax3.annotate(
    "",
    xy=(mu2, sigma2),
    xytext=(mu1, sigma1),
    arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5),
)
#add text to middle of the arrow

ax3.text(
    (mu1 + mu2) / 2 + 0.3,
    (sigma1 + sigma2) / 2,
    r"$D$",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
)"""


plt.show()


plt.show()

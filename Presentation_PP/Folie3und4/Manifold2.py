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
mu = 0
sigma = 0.9
# ax1.plot(x, gaussian(x, mu, sigma), color="red", linewidth=2)
# ax2.plot(mu, sigma, "o", color="red", linewidth=2)

plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import numpy as np

# Enable latex text rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
# increase font size
plt.rcParams.update({"font.size": 16})

# Explain gradient descent

# lets make a spherical plot
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

"""
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)


def gauss(x, y, mx, my, sx, sy, Amplitude, C):
    return (
        -Amplitude * np.exp(-1 / sx * (x - mx) ** 2) * np.exp(-1 / sy * (y - my) ** 2)
        + C
    )


def Z_fn(X, Y):
    return (
        gauss(X, Y, 0, 0, 40, 40, 1, 2)
        + gauss(X, Y, 1, 1, 2, 2, -3, 1)
        + gauss(X, Y, -2, 1, 5, 2, -1.5, 1)
        + gauss(X, Y, -3, 0.5, 1, 1.3, -0.2, 1)
        + gauss(X, Y, -2, -1, 2, 2, 1, 1)
        + gauss(X, Y, 1, -1, 2, 2, -1, 0.5)
        + gauss(X, Y, -4, -4, 10, 10, 1.3, 1)
        + gauss(X, Y, -0.37, 1.7, 2, 2, -0.1, 0)
    )


def grad_fn(X, Y):
    e = 1e-4
    xgrad = (Z_fn(X + e, Y) - Z_fn(X - e, Y)) / 2 / e
    ygrad = (Z_fn(X, Y + e) - Z_fn(X, Y - e)) / 2 / e
    return np.array([xgrad, ygrad])


F = Z_fn(X, Y)
"""


R = np.arange(0, 3, 0.01)
phi = np.arange(0, 2 * np.pi, 0.01)
R, phi = np.meshgrid(R, phi)


def Z_fn(R, phi):
    return 0.5 * ((R * np.cos(phi) - 1) ** 2 + (R * np.sin(phi)) ** 2)


def grad_fn(X, Y):
    e = 1e-6
    xgrad = (Z_fn(X + e, Y) - Z_fn(X - e, Y)) / 2 / e
    ygrad = (Z_fn(X, Y + e) - Z_fn(X, Y - e)) / 2 / e
    return np.array([xgrad, ygrad])


def grad_fn_polar(R, phi):
    Rgrad = (R * np.cos(phi) - 1) * np.cos(phi) + R * np.sin(phi) * np.sin(phi)
    phigrad = (
        1
        / R
        * ((R * np.cos(phi) - 1) * (-R * np.sin(phi)) + R * np.sin(phi) * np.cos(phi))
    )
    return np.array([Rgrad, phigrad])


def gradient_descent(startx, starty, learning_rate, n_steps):
    xlist, ylist, zlist = (
        np.array([startx]),
        np.array([starty]),
        np.array([Z_fn(startx, starty) + 0.1]),
    )
    currentx, currenty = startx, starty

    for i in range(n_steps):
        currentgrad = grad_fn(currentx, currenty)
        currentx -= learning_rate * currentgrad[0]
        currenty -= learning_rate * currentgrad[1]
        xlist = np.append(xlist, currentx)
        ylist = np.append(ylist, currenty)
        zlist = np.append(zlist, Z_fn(currentx, currenty) + 0.1)
    return xlist, ylist, zlist


def polar_gradient_descent(startR, startphi, learning_rate, n_steps):
    Rlist, philist, zlist = (
        np.array([startR]),
        np.array([startphi]),
        np.array([Z_fn(startR, startphi) + 0.1]),
    )
    currentR, currentphi = startR, startphi

    for i in range(n_steps):
        currentgrad = grad_fn_polar(currentR, currentphi)
        currentR -= learning_rate * currentgrad[0]
        currentphi -= learning_rate * currentgrad[1]
        Rlist = np.append(Rlist, currentR)
        philist = np.append(philist, currentphi)
        zlist = np.append(zlist, Z_fn(currentR, currentphi) + 0.1)
    return Rlist, philist, zlist


##################################
##################################
##################################
# Here the code starts


Rlist, philist, zlist = gradient_descent(2.7, 1.7, 0.3, 20)
Rlist2, philist2, zlist2 = polar_gradient_descent(2.7, 1.7, 0.3, 20)


F = Z_fn(R, phi)


contour = plt.contour(
    phi,
    R,
    F,
    [0, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7],
    cmap=cm.magma,
)
plt.plot(philist[0], Rlist[0], ".-", color="#00E88F", linewidth=2, markersize=20)
plt.plot(philist, Rlist, ".-", color="#00E88F", linewidth=2, markersize=10)

# Add radial arrow
arrow_length_r = 2 * np.pi - 0.01
arrow_angle_r = np.pi / 3  # Angle of the arrow
ax.annotate(
    "",
    xy=(arrow_angle_r, arrow_length_r),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=1.8, shrinkA=0, shrinkB=0),
    transform=ax.transData,
)
ax.text(
    arrow_angle_r,
    arrow_length_r + 0.1,
    r"$r$",
    fontsize=12,
    ha="center",
    transform=ax.transData,
)

# remove radial tick labels
ax.set_yticklabels([])

# Set xticks to 0, pi/2, pi, 3pi/2
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
# Format angular tick labels as radians
ax.set_xticklabels(
    [
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
    ]
)


# Add coordinate grid
ax.grid(True)


plt.show()
plt.close()

# create figure and axis with specific axis size
fig, ax = plt.subplots()
img = ax.imshow(
    np.transpose(F)[::-1, :], cmap=cm.magma, extent=[0, 2 * np.pi, 0, 3], alpha=0
)
plt.contour(phi, R, F, [0, 0.2, 0.4, 0.6, 1, 2, 3, 4, 5, 6, 7], cmap=cm.magma)
cbar = plt.colorbar(img, fraction=0.025, pad=0.04)
cbar.solids.set(alpha=1)
cbar.set_label("Potential $V$")
ax.plot(philist[0], Rlist[0], ".-", color="#00E88F", linewidth=2, markersize=20)
ax.plot(philist2, Rlist2, ".-", color="#00E88F", linewidth=2, markersize=10)

# add labels
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$r$")

# Set xticks to 0, pi/2, pi, 3pi/2
"""
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
# Format angular tick labels as radians
ax.set_xticklabels(
    [
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
    ]
)
"""

ax.grid(True)
plt.show()

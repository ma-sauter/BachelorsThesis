import matplotlib.pyplot as plt
import numpy as np

# enable latex text rendering with a serif font and font size 16
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 16})


fig = plt.figure()
ax1 = fig.add_subplot(121, polar=True)
# adjust figure size
fig.set_size_inches(10, 5)
# add horizontal space between subplots
fig.subplots_adjust(wspace=1)

ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(0, 3)
# remove radial labels
ax1.set_yticklabels([])

# now create an empty plot in regular coordinates
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, 2 * np.pi)
ax2.set_ylim(0, 3)
ax2.set_xlabel(r"$\phi$")
ax2.set_ylabel(r"$r$")
# write phi labels in radians
ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax2.set_xticklabels(
    [
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
    ]
)
ax2.grid(True)

# Now add some lines
height = 0.5
phi = np.linspace(np.pi / 2, np.pi, 100)
r = np.ones_like(phi) * height
ax1.plot(phi, r, color="red", linewidth=2)
ax2.plot(phi, r, color="red", linewidth=2)


plt.show()

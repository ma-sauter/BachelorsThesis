import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_box_aspect((4,4,1))

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

def gauss(x,y, mx, my, sx, sy, Amplitude, C):
    return -Amplitude*np.exp(-1/sx*(x-mx)**2)*np.exp(-1/sy*(y-my)**2) +C

def Z_fn(X,Y):
    return (gauss(X,Y,0,0,40,40,1,2) + gauss(X,Y,1,1,2,2,-3,1) + gauss(X,Y,-2,1,5,2,-1.5,1) + gauss(X,Y,-3,0.5,1,1.3,-0.2,1) +
            gauss(X,Y,-2,-1,2,2,1,1) + gauss(X,Y,1,-1,2,2,-1,0.5) + gauss(X,Y,-4,-4,10,10,1.3,1) + gauss(X,Y,-0.37,1.7,2,2,-0.1,0))
def grad_fn(X,Y):
    e = 1e-4
    xgrad = (Z_fn(X+e,Y) - Z_fn(X-e,Y))/2/e
    ygrad = (Z_fn(X,Y+e) - Z_fn(X,Y-e))/2/e
    return np.array([xgrad,ygrad])

F = Z_fn(X,Y)


def gradient_descent(startx,starty, learning_rate, n_steps):
    xlist, ylist, zlist = np.array([startx]), np.array([starty]), np.array([Z_fn(startx,starty)+0.1])
    currentx, currenty = startx, starty

    for i in range(n_steps):
        currentgrad = grad_fn(currentx,currenty)
        currentx -= learning_rate*currentgrad[0]
        currenty -= learning_rate*currentgrad[1]
        xlist = np.append(xlist, currentx)
        ylist = np.append(ylist, currenty)
        zlist = np.append(zlist, Z_fn(currentx,currenty)+0.1)
    return xlist, ylist, zlist



colormap = plt.get_cmap("BuGn")
color1 = colormap(0.8)
colormap = plt.get_cmap("cool")
color2 = colormap(0.2)
color1 = color2

# Plot the surface.
surf = ax.plot_surface(X, Y, F, cmap=cm.magma,
                       linewidth=0, antialiased=True)
d1x, d1y, d1z = gradient_descent(0.3,0.95,1.5,5)
d2x, d2y, d2z = gradient_descent(0.3,0.95,0.01,1000)
des1 = ax.plot(d1x,d1y,d1z, "o--", color = color1, markersize = 7, zorder = 10)
des2 = ax.plot(d2x,d2y,d2z, color = color2,  zorder = 9)

# Customize the z axis.
ax.set_zlim(5.3, 8.5)
value = 0.4
ax.set_xlim(-3.8-value,4.4+value)
ax.set_ylim(-3.8-value,3.8+value)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.view_init(elev=24., azim=-110.)
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')


ax.set_axis_off()

plt.tight_layout()
#plt.show()
plt.savefig("GradientFlowPlot.pdf", bbox_inches='tight')
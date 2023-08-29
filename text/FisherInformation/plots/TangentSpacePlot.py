import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm

ax = plt.axes(projection='3d',computed_zorder=False)
a,b = 0.48,0.1
l = 0.2

X = np.linspace(0,0.75,100)
Y = np.linspace(-1,1,100)
X,Y = np.meshgrid(X,Y)
X2 = np.linspace(a-l,a+l,7)
Y2 = np.linspace(b-l,b+l,7)
X2,Y2 = np.meshgrid(X2,Y2)

def Zfunc(x,y):
    return np.sin(4*x)*np.exp(-5*y**2)

def tangent_space(x,y, a, b):
    e = 1e-3
    d1 = (Zfunc(a+e,b)-Zfunc(a-e,b))/2/e
    d2 = (Zfunc(a,b+e)-Zfunc(a,b-e))/2/e

    return Zfunc(a,b)+d1*(x-a)+d2*(y-b)

Z = Zfunc(X,Y)
Z2 = tangent_space(X2,Y2,a,b)

ax.plot_surface(X2,Y2,Z2,zorder = 500, edgecolor = '#00E88F', alpha = 0, antialiased = True)

ax.plot_surface(X,Y,Z, cmap = cm.magma,
                antialiased=True, zorder = 1, edgecolor = "black", linewidth = 0.1)
ax.plot(a,b,Zfunc(a,b), "o", color = '#00E88F')

ax.view_init(elev=20., azim=113.)
ax.set_axis_off()

#plt.show()
plt.savefig("TangentSpacePlot.pdf", bbox_inches='tight')
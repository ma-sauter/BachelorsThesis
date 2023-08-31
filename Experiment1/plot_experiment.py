import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

PLOTLOSSSURFACE = True
PLOTFISHERSURFACE = True

if PLOTLOSSSURFACE:
    X = np.load("loss_surf_plot.npz")['X']
    Y = np.load("loss_surf_plot.npz")['Y']
    Z = np.load("loss_surf_plot.npz")['Z']
    t_list,l_list,acc = np.load("training.npy")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                        linewidth=0, antialiased=True)
    path = ax.plot(t_list[0],t_list[1], l_list, "-",color = 'mediumseagreen', zorder=10)

    plt.title("loss surface and training evolution")
    plt.show()
    plt.close()

if PLOTFISHERSURFACE:
    t1,t2 = 1,1
    X,Y,Z,t_list,pathZ = np.load(f"Fisher_surf_plot{t1}{t2}.npy")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=True)
    path = ax.plot(t_list[0],t_list[1],Z2, color = 'mediumseagreen', zorder=100)

    ax.view_init(elev=90., azim=0.)

    plt.title("F{t1}{t2} surface")
    plt.show()
    plt.close()

if PLOTCURVESURFACE:
    X,Y,Z,t_list,pathZ = np.load(f"curvature_plot.npy")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma,
                       linewidth=0, antialiased=True)
    path = ax.plot(t_list[0],t_list[1],pathZ, color = 'mediumseagreen', zorder=100)

    ax.view_init(elev=90., azim=0.)

    plt.title("Scalar curvature surface")
    plt.show()
    plt.close()
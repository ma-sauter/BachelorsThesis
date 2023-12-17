import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


import matplotlib.pyplot as plt

lossname_for_fisher_and_ntk = "MeanPowerLoss2"


def create_animation(lossname):
    print("Creating GIF for ", lossname)
    data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    t_list = data["t_list"]
    Zpath = data["Zpath"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # add axis labels
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_zlabel(r"$R$")
    # lock the label rotation to 0 degrees
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)

    # Create the initial plot
    plot = ax.plot_surface(X, Y, Z, cmap="magma")
    # also add path
    path = ax.plot(
        t_list[0][::20], t_list[1][::20], Zpath, color="mediumseagreen", zorder=100
    )

    if lossname == "MeanPowerLoss2":
        # set axis limits
        ax.set_zlim(-1000, 500)
    if lossname == "LPNormLoss2":
        ax.set_zlim(-200, 0)
    if lossname == "CrossEntropyLoss":
        ax.set_zlim(-100, 0)

    # add title of lossname
    # ax.set_title(lossname)

    def update(frame):
        # Rotate the plot around the Z-axis
        ax.view_init(elev=20, azim=frame)

    # Create the animation
    animation = FuncAnimation(
        fig, update, frames=np.arange(0, 360, 1.5), interval=1000 / 120
    )

    # save the animation as a mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    animation.save(
        f"GIFS/curvature_{lossname}.mp4",
        fps=30,
        extra_args=["-vcodec", "libx264"],
        dpi=400,
    )
    plt.close()


# ----------------------------------------------------------------

# Create Fisher Trace and NTK Trace plots


def create_trace_animation(lossname):
    print("Creating trace GIF for ", lossname)
    data = np.load(f"npfiles/{lossname}_Fisher_infos.npz", allow_pickle=True)
    X_Fisher, Y_Fisher = data["X"], data["Y"]
    Z_Fisher = data[f"Z11"] + data[f"Z22"]
    t_list_Fisher = data["t_list"]
    Zpath_Fisher = data[f"Zpath11"] + data[f"Zpath22"]

    data = np.load(f"npfiles/{lossname}_ntk.npz")
    t_list_NTK = data["t_list"]
    Zpath_NTK = data["NTK_path"]
    X_NTK = data["X"]
    Y_NTK = data["Y"]
    Z_NTK = data["Z"]

    fig = plt.figure()
    ax_Fisher = fig.add_subplot(121, projection="3d")
    ax_NTK = fig.add_subplot(122, projection="3d")

    # Create the initial plot
    plot_Fisher = ax_Fisher.plot_surface(X_Fisher, Y_Fisher, Z_Fisher, cmap="magma")
    plot_NTK = ax_NTK.plot_surface(X_NTK, Y_NTK, Z_NTK, cmap="magma")

    # also plot paths
    path_Fisher = ax_Fisher.plot(
        t_list_Fisher[0],
        t_list_Fisher[1],
        Zpath_Fisher,
        color="mediumseagreen",
        zorder=100,
    )
    path_NTK = ax_NTK.plot(
        t_list_NTK[0], t_list_NTK[1], Zpath_NTK, color="mediumseagreen", zorder=100
    )

    # add title of traces
    ax_Fisher.set_title("Fisher Trace")
    ax_NTK.set_title("NTK Trace")
    # add axis labels
    ax_Fisher.set_xlabel(r"$\theta_1$")
    ax_Fisher.set_ylabel(r"$\theta_2$")
    # lock the label rotation to 0 degrees
    ax_Fisher.xaxis.set_rotate_label(False)
    ax_Fisher.yaxis.set_rotate_label(False)
    # Do the same for NTK
    ax_NTK.set_xlabel(r"$\theta_1$")
    ax_NTK.set_ylabel(r"$\theta_2$")
    # lock the label rotation to 0 degrees
    ax_NTK.xaxis.set_rotate_label(False)
    ax_NTK.yaxis.set_rotate_label(False)

    def update(frame):
        # Rotate the plot around the Z-axis
        ax_Fisher.view_init(elev=20, azim=frame)
        ax_NTK.view_init(elev=20, azim=frame)

    animation = FuncAnimation(
        fig, update, frames=np.arange(0, 360, 1.5), interval=1000 / 120
    )

    # save the animation as a mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    animation.save(
        f"GIFS/traces_{lossname}.mp4",
        fps=30,
        extra_args=["-vcodec", "libx264"],
        dpi=400,
    )
    plt.close()


# Actual creation of GIFs

for lossname in ["LPNormLoss2", "MeanPowerLoss2", "CrossEntropyLoss"]:
    create_animation(lossname)
    create_trace_animation(lossname)

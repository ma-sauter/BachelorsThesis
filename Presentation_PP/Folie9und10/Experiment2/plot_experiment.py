import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
import plotly.offline as pyo

lossname = "MeanPowerLoss2"
PLOTFUNCTION = False
PLOTLOSSSURFACE = False
PLOTFISHERSURFACE = False
PLOTFISHERSURFACEPLOTLY = False
t1, t2 = 2, 2
PLOTCURVESURFACE = False
PLOTCURVESURFACEPLOTLY = False
PLOTLONGTRAINING = False
PLOTNTKSURFACE = False

PLOTCURVEFORPRESENATION = True

if PLOTFUNCTION:
    data = np.load(f"npfiles/{lossname}_training.npz")
    X, Y, Z = data["Xfunc"], data["Yfunc"], data["Zfunc"]
    print(data["t_list"])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cm.magma)
    plt.show()


if PLOTLOSSSURFACE:
    data = np.load(f"npfiles/{lossname}_training.npz")
    X = data["SurfX"]
    Y = data["SurfY"]
    Z = data["SurfZ"]
    t_list = data["t_list"]
    l_list = data["l_list"]
    acc = data["acc_list"]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(t_list[0], t_list[1], l_list, "-", color="mediumseagreen", zorder=10)

    plt.title("loss surface and training evolution")
    plt.show()
    plt.close()
    plt.plot(acc)
    plt.title("accuracy")
    plt.show()

if PLOTFISHERSURFACE:
    data = np.load(f"npfiles/{lossname}_Fisher_infos.npz", allow_pickle=True)
    X, Y = data["X"], data["Y"]
    Z = data[f"Z{t1}{t2}"]
    t_list = data["t_list"]
    Zpath = data[f"Zpath{t1}{t2}"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(t_list[0], t_list[1], Zpath, color="mediumseagreen", zorder=100)

    ax.view_init(elev=90.0, azim=0.0)

    plt.title(f"F{t1}{t2} surface")
    plt.show()
    plt.close()

if PLOTFISHERSURFACEPLOTLY:
    # Load data
    data = np.load(f"npfiles/{lossname}_Fisher_infos.npz", allow_pickle=True)
    X, Y = data["X"], data["Y"]
    Z = data[f"Z{t1}{t2}"]
    t_list = data["t_list"]
    Zpath = data[f"Zpath{t1}{t2}"]

    # Create surface plot
    surface_trace = go.Surface(x=X, y=Y, z=Z, colorscale="magma")

    # Create path trace
    path_trace = go.Scatter3d(
        x=t_list[0],
        y=t_list[1],
        z=pathZ,
        mode="lines",
        line=dict(color="mediumseagreen", width=4),
        name="Path",
    )

    # Create layout
    layout = go.Layout(
        scene=dict(
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.2),
        ),
        title=f"F{t1}{t2} surface",
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=[surface_trace, path_trace], layout=layout)

    # Save the figure as an interactive HTML file
    html_filename = f"Interactive_Fisher{t1}{t2}.html"
    pyo.plot(fig, filename=html_filename, auto_open=True)


if PLOTCURVESURFACE:
    data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    t_list = data["t_list"]
    Zpath = data["Zpath"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(
        t_list[0][::20], t_list[1][::20], Zpath, color="mediumseagreen", zorder=100
    )
    end = ax.plot(
        t_list[0][-1], t_list[1][-1], Zpath[-1], "X", color="mediumseagreen", zorder=100
    )

    # ax.set_zlim(-0.2e9, 0.2e9)

    ax.view_init(elev=90.0, azim=0.0)

    plt.title("Scalar curvature surface")
    plt.show()
    plt.close()

if PLOTCURVESURFACEPLOTLY:
    # Load data
    data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    X, Y, Z, t_list, Zpath = (
        data["X"],
        data["Y"],
        data["Z"],
        data["t_list"],
        data["Zpath"],
    )
    # Create surface trace
    surface_trace = go.Surface(x=X, y=Y, z=Z, colorscale="magma")

    # Create path trace
    path_trace = go.Scatter3d(
        x=t_list[0][::20],
        y=t_list[1][::20],
        z=Zpath,
        mode="lines",
        line=dict(color="mediumseagreen", width=4),  # Adjust line width as needed
        name="Path",
    )

    cross = go.Scatter3d(
        x=[t_list[0][-1]],
        y=[t_list[1][-1]],
        z=[Zpath[-1]],
        mode="markers",
        marker=dict(symbol="circle", color="mediumseagreen", size=5, z=100),
        name="End",
    )

    # Create layout
    layout = go.Layout(
        scene=dict(
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.2),
            zaxis=dict(),  # range=[-0.2e9, 0.2e9]),  # Set z-axis limits
        ),
        title="Scalar curvature surface",
    )

    # Combine traces and layout into a figure
    fig = go.Figure(data=[surface_trace, path_trace, cross], layout=layout)

    # Save the figure as an interactive HTML file
    html_filename = "curvature_plot.html"
    pyo.plot(fig, filename=html_filename, auto_open=True)

if PLOTLONGTRAINING:
    data = np.load(f"npfiles/{lossname}_long_training.npz")
    t_list = data["t_list"]
    c_list = data["c_list"]
    NTK_trace = data["NTK_trace_list"]
    Fisher_trace = data["Fisher_trace_list"]

    plt.plot(c_list)
    plt.title("Curvature")
    plt.show()
    plt.close()

    plt.plot(NTK_trace)
    plt.title("NTK trace")
    plt.show()
    plt.close()

    plt.plot(Fisher_trace)
    plt.title("Fisher trace")
    plt.show()
    plt.close()

if PLOTNTKSURFACE:
    data = np.load(f"npfiles/{lossname}_ntk.npz")
    t_list = data["t_list"]
    Zpath = data["NTK_path"]
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(t_list[0], t_list[1], Zpath, color="mediumseagreen", zorder=100)
    plt.show()

if PLOTCURVEFORPRESENATION:
    data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    t_list = data["t_list"]
    Zpath = data["Zpath"]

    # now make an animation of the plot rotating
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
import matplotlib.pyplot as plt

# ... (previous code)

if PLOTCURVEFORPRESENATION:
    data = np.load(f"npfiles/{lossname}curvature_plot.npz")
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    t_list = data["t_list"]
    Zpath = data["Zpath"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create the initial plot
    plot = ax.plot_surface(X, Y, Z, cmap="magma")

    # add title of lossname
    ax.set_title(lossname)

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

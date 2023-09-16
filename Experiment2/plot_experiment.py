import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo


PLOTLOSSSURFACE = False
PLOTFISHERSURFACE = True
PLOTFISHERSURFACEPLOTLY = False
t1, t2 = 2, 2
PLOTCURVESURFACE = False
PLOTCURVESURFACEPLOTLY = False

if PLOTLOSSSURFACE:
    X = np.load("loss_surf_plot.npz")["X"]
    Y = np.load("loss_surf_plot.npz")["Y"]
    Z = np.load("loss_surf_plot.npz")["Z"]
    t_list = np.load("training.npz")["t_list"]
    l_list = np.load("training.npz")["l_list"]
    acc = np.load("training.npz")["acc"]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(t_list[0], t_list[1], l_list, "-", color="mediumseagreen", zorder=10)

    plt.title("loss surface and training evolution")
    plt.show()
    plt.close()

if PLOTFISHERSURFACE:
    data = np.load("npfiles/Fisher_infos.npz", allow_pickle=True)
    X, Y = data["X"], data["Y"]
    Z = data[f"Z{t1}{t2}"]
    t_list = data["t_list"]
    Zpath = data[f"Zpath{t1}{t2}"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    path = ax.plot(
        t_list[0][::20], t_list[1][::20], Zpath, color="mediumseagreen", zorder=100
    )

    ax.view_init(elev=90.0, azim=0.0)

    plt.title(f"F{t1}{t2} surface")
    plt.show()
    plt.close()

if PLOTFISHERSURFACEPLOTLY:
    # Load data
    X, Y, Z, t_list, pathZ = np.load(f"Fisher_surf_plot{t1}{t2}.npy", allow_pickle=True)

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
    X = np.load(f"npfiles/curvature_plot.npz")["X"]
    Y = np.load(f"npfiles/curvature_plot.npz")["Y"]
    Z = np.load(f"npfiles/curvature_plot.npz")["Z"]
    t_list = np.load(f"npfiles/curvature_plot.npz")["t_list"]
    Zpath = np.load(f"npfiles/curvature_plot.npz")["Zpath"]
    print(Zpath)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, linewidth=0, antialiased=True)
    # path = ax.plot(
    #    t_list[0][::20], t_list[1][::20], Zpath, color="mediumseagreen", zorder=100
    # )
    # end = ax.plot(
    #    t_list[0][-1], t_list[1][-1], Zpath, "X", color="mediumseagreen", zorder=100
    # )

    # ax.set_zlim(-0.2e9, 0.2e9)

    ax.view_init(elev=90.0, azim=0.0)

    plt.title("Scalar curvature surface")
    plt.show()
    plt.close()

if PLOTCURVESURFACEPLOTLY:
    # Load data
    data = np.load("npfiles/curvature_plot.npz")
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

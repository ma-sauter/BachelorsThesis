import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plotly.graph_objects as go

import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
from Curvature_calculation import curvature_slow_but_working as curvature
from rich.progress import track
import pickle
import time


from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)


CALCULATE_ISING_CURVATURE = True
PLOT_ISING_CURVATURE = False
PLOTLY = True

## Import dataset
with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)


## Define network
def network():
    return None


## Define subloss
def subloss(input, target, theta, network):
    n_part = 10
    J = 0.01
    beta, H = theta[0], theta[1]
    lambda1 = np.exp(beta * J) * (
        np.cosh(beta * H) + np.sqrt(np.sinh(beta * H) ** 2 + np.exp(-4 * beta * J))
    )
    lambda2 = np.exp(beta * J) * (
        np.cosh(beta * H) - np.sqrt(np.sinh(beta * H) ** 2 + np.exp(-4 * beta * J))
    )
    return lambda1**n_part + lambda2**n_part


## Calculation of curvature for ising model
if CALCULATE_ISING_CURVATURE:
    theta1 = np.linspace(0.1, 2.3, 5)
    theta2 = np.linspace(-1, 1.6, 5)
    X, Y = np.meshgrid(theta1, theta2)

    Z = onp.zeros_like(X)
    progress = -1
    for i, theta1_ in enumerate(theta1):
        for j in range(len(theta2)):
            progress += 1
            print(
                f"Calculating scalar curvatures done {100*progress/len(theta1)/len(theta2)}%"
            )
            Z[j, i] = curvature(
                subloss, network, dataset, theta=np.array([theta1[i], theta2[j]])
            )
            print(Z[j, i])

    np.savez("npfiles/ising_curv.npz", X=X, Y=Y, Z=Z, allow_pickle=True)

if PLOT_ISING_CURVATURE:
    data = np.load("npfiles/ising_curv.npz")
    X, Y, Z = data["X"], data["Y"], np.fabs(data["Z"])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(np.exp(X), np.exp(Y), Z, cmap=cm.magma)
    ax.set_zscale("log")
    ax.set_zlim(-0, 10000)
    plt.show()
    print(Z)

if PLOTLY:
    # Load data
    data = np.load("npfiles/ising_curv.npz")
    X, Y, Z = data["X"], data["Y"], np.fabs(data["Z"])
    Z = np.log10(Z)
    # Create a 3D surface plot
    fig = go.Figure(
        data=[go.Surface(z=Z, x=np.exp(X), y=np.exp(Y), colorscale="magma")]
    )

    # Set the z-scale to logarithmic
    # fig.update_scenes(zaxis_type="log")

    # Set z-axis limits
    fig.update_layout(scene=dict(zaxis=dict(range=[-16, 15])))

    # Show the plot
    fig.show()

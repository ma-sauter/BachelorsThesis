import matplotlib.pyplot as plt
from matplotlib import cm
import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
from Curvature_calculation import curvature
from rich.progress import track
import pickle
import time


from jax.lib import xla_bridge

CUDA_VISIBLE_DEVICES = 0
print(xla_bridge.get_backend().platform)


CALCULATE_ISING_CURVATURE = True
PLOT_ISING_CURVATURE = False

## Import dataset
with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)


## Define network
def network():
    return None


## Define subloss
def subloss(input, target, theta, network):
    n_part = 10
    J = 1
    H = 1
    beta, h = theta[0], theta[1]
    lambda1 = np.exp(beta * J) * (
        np.cos(beta * H) + np.sqrt(np.sinh(beta * H) ** 2 - np.exp(-4 * beta * J))
    )
    lambda2 = np.exp(beta * J) * (
        np.cos(beta * H) - np.sqrt(np.sinh(beta * H) ** 2 - np.exp(-4 * beta * J))
    )
    return lambda1**n_part + lambda2**n_part


## Calculation of curvature for ising model
if CALCULATE_ISING_CURVATURE:
    theta1 = np.linspace(0, 2.3, 10)
    theta2 = np.linspace(-1, 1.6, 10)
    X, Y = np.meshgrid(theta1, theta2)

    Z = onp.zeros_like(X)
    for i, theta1_ in enumerate(theta1):
        print(f"Calculating scalar curvatures done {i}%")
        for j in track(range(len(theta2))):
            Z[j, i] = curvature(
                subloss, network, dataset, theta=np.array([theta1[i], theta2[j]])
            )

    np.savez("npfiles/ising_curv.npz", X=X, Y=Y, Z=Z, allow_pickle=True)

if PLOT_ISING_CURVATURE:
    data = np.load("npfiles/ising_curv.npz")
    X, Y, Z = data["X"], data["Y"], data["Z"]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(np.exp(X), np.exp(Y), Z, cmap=cm.magma)
    plt.show()

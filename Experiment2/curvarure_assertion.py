import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plotly.graph_objects as go

import jax.numpy as np
import jax
from jax import grad, jacfwd, debug, vmap as jvmap
import numpy as onp
from rich.progress import track
import pickle
import time


# Curvature with hessians
def curvature(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    hessian = np.array(
        jacfwd(jacfwd(fisher_info, argnums=3), argnums=3)(
            subloss, network, dataset, theta
        )
    )

    n = len(theta)
    R = 0

    def vmap_func(mu, v, alpha, beta, ig, hessian):
        return (
            ig[beta, v]
            * ig[alpha, mu]
            * 0.5
            * (
                hessian[mu][beta][alpha, v]
                - hessian[v][beta][alpha, mu]
                + hessian[v][alpha][beta, mu]
                - hessian[mu][alpha][beta, v]
            )
        )

    vmap1 = jvmap(vmap_func, in_axes=(0, None, None, None, None, None))
    vmap2 = jvmap(vmap1, in_axes=(None, 0, None, None, None, None))
    vmap3 = jvmap(vmap2, in_axes=(None, None, 0, None, None, None))
    vmap = jvmap(vmap3, in_axes=(None, None, None, 0, None, None))

    pars = np.arange(len(theta))
    Rlist = vmap(pars, pars, pars, pars, ig, hessian)
    # debug.print("{x}", x=hessian)
    return np.sum(Rlist)


# Curvature with christoffels
def curvature1(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    fisher_derivatives = jacfwd(fisher_info, argnums=3)(
        subloss, network, dataset, theta
    )

    def christoffel(i, j, k):
        symbol = 0
        i, j, k = int(i), int(j), int(k)
        for m in range(len(theta)):
            symbol += ig[m, i] * (
                fisher_derivatives[k][m, j]
                + fisher_derivatives[j][m, k]
                - fisher_derivatives[m][j, k]
            )
        return 0.5 * symbol

    # Derivatives of the christoffel symbol
    dChristoffel = [grad(christoffel, 0), grad(christoffel, 1), grad(christoffel, 2)]

    curvature = 0
    n_t = len(theta)

    for i in range(n_t):
        for j in range(n_t):
            for m in range(n_t):
                for n in range(n_t):
                    curvature += ig[i, j] * (
                        dChristoffel[m](m * 1.0, i * 1.0, j * 1.0)
                        - dChristoffel[j](m * 1.0, i * 1.0, m * 1.0)
                        + christoffel(n, i, j) * christoffel(m, m, n)
                        - christoffel(n, i, m) * christoffel(m, j, n)
                    )
    return curvature


def fisher_info(subloss, network, dataset, theta):
    return np.array([[np.exp(theta[1]), 0], [0, 1]])


## My curvature calculation
def subloss():
    return None


def network():
    return None


dataset = np.array([])


assert curvature(subloss, network, dataset, theta=[0.0, 5.0]) == -1
assert curvature(subloss, network, dataset, theta=[2.0, 1.0]) == -1
assert curvature(subloss, network, dataset, theta=[5.0, 0.0]) == -1
assert curvature(subloss, network, dataset, theta=[1.0, 5.0]) == -1
assert curvature(subloss, network, dataset, theta=[7.0, 10000.0]) == -1 or np.nan
assert curvature(subloss, network, dataset, theta=[10000.0, -8.0]) == -1


def fisher_info(subloss, network, dataset, theta):
    t, r, teta, phi = theta[0], theta[1], theta[2], theta[3]
    return np.array(
        [
            [(1 - 2 / r), 0, 0, 0],
            [0, -1 / (1 - 2 / r), 0, 0],
            [0, 0, -(r**2), 0],
            [0, 0, 0, -(r**2) * np.sin(teta) ** 2],
        ]
    )


print(curvature(subloss, network, dataset, theta=[1.0, 0.5, 2.0, 2.0]))

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
            ig[mu, v]
            * ig[alpha, beta]
            * (
                hessian[mu][beta][alpha, v]
                - hessian[v][beta][alpha, mu]
                + hessian[v][alpha][beta, mu]
                - hessian[mu][alpha][alpha, mu]
            )
        )

    vmap1 = jvmap(vmap_func, in_axes=(0, None, None, None, None, None))
    vmap2 = jvmap(vmap1, in_axes=(None, 0, None, None, None, None))
    vmap3 = jvmap(vmap2, in_axes=(None, None, 0, None, None, None))
    vmap = jvmap(vmap3, in_axes=(None, None, None, 0, None, None))

    pars = np.arange(len(theta))
    Rlist = vmap(pars, pars, pars, pars, ig, hessian)
    return np.sum(Rlist) / 2


def fisher_info(subloss, network, dataset, theta):
    return np.array([[theta[0] * theta[1], 0], [0, theta[0] * theta[1]]])


import sympy
from sympy import pprint, symbols, diag, sin, cos, exp
from einsteinpy.symbolic import RicciTensor, RicciScalar
from einsteinpy.symbolic.metric import MetricTensor
from einsteinpy.symbolic.predefined import AntiDeSitter

sympy.init_printing()

# Define the symbols
t, r, theta, phi = symbols("t r theta phi")

# Define the metric for Minkowski space
metric = sympy.Array([[t * r, 0], [0, t * r]])

# Create the metric tensor
g = MetricTensor(metric, (t, r))

Ric = RicciTensor.from_metric(g)
Ric.tensor()

R = RicciScalar.from_riccitensor(Ric)
R.simplify()
pprint(f"analytic curvature {R.expr}")


## My curvature calculation
def subloss():
    return None


def network():
    return None


dataset = np.array([])

print(curvature(subloss, network, dataset, theta=[1.0, 2.0]))

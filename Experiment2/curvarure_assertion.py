import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plotly.graph_objects as go

import jax.numpy as np
import jax
from jax import grad, jacfwd, debug
import numpy as onp
from rich.progress import track
import pickle
import time
import sympy
from sympy import pprint, symbols, diag, sin, cos, exp
from einsteinpy.symbolic import RicciTensor, RicciScalar
from einsteinpy.symbolic.metric import MetricTensor
from einsteinpy.symbolic.predefined import AntiDeSitter

sympy.init_printing()

# Define the symbols
t, r, theta, phi = symbols("t r theta phi")

# Define the metric for Minkowski space
metric = sympy.Array([[t, 0], [0, t * r**2]])

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

print(curvature(subloss, network, dataset, [0.1, 2.0]))
print(curvature(subloss, network, dataset, [0.1, 1.0]))

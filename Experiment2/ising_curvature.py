import matplotlib.pyplot as plt
import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
from Curvature_calculation import curvature
from rich.progress import track
import pickle
import time

CALCULATE_ISING_CURVATURE = True

## Import Dataset
dataset = np.array([])

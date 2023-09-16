import matplotlib.pyplot as plt
import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
from rich.progress import track
import pickle
import time

CALCULATE_SCALAR_CURVATURE = True

## Import dataset
with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)

## Define Network
from networks import OneNode_DB_network

network = OneNode_DB_network.network

## Define Loss functions
from losses import MeanPowerLoss2

loss = MeanPowerLoss2.loss
subloss = MeanPowerLoss2.subloss


def test_implementation():
    curvature(subloss, network, dataset, theta=np.array([0.0, 5.0]))
    start1 = time.time()
    curvature(subloss, network, dataset, theta=np.array([0.0, 5.0]))
    end1 = time.time()
    start2 = time.time()
    curvature(subloss, network, dataset, theta=np.array([1.0, 1.0]))
    end2 = time.time()
    start3 = time.time()
    curvature(subloss, network, dataset, theta=np.array([3.0, 0.0]))
    end3 = time.time()
    print(f"This took {(end1+end2+end3-start1-start2-start3)/3} seconds")


from Curvature_calculation import curvature2_vmap as curvature

curvature(subloss, network, dataset, theta=np.array([0.0, 5.0]))
print("Using vmap implementation")
test_implementation()

from Curvature_calculation import curvature2 as curvature

print("Using implementation without vmaps")
test_implementation()

from Curvature_calculation import curvature_slow_but_working as curvature

print("Using working but unoptimized christoffel version")
test_implementation()

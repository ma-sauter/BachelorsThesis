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


if CALCULATE_SCALAR_CURVATURE:
    # Scalar Curvature

    theta1 = np.linspace(-2, 1, 100)
    theta2 = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(theta1, theta2)
    t_list = np.load("npfiles/training.npz")["t_list"]
    l_list = np.load("npfiles/training.npz")["l_list"]
    acc = np.load("npfiles/training.npz")["acc"]

    Z = onp.zeros_like(X)
    for i, theta1_ in enumerate(theta1):
        print(f"Calculating scalar curvatures done {i}%")
        for j in track(range(len(theta2))):
            Z[j, i] = curvature(
                subloss, network, dataset, theta=np.array([theta1[i], theta2[i]])
            )

    Zpath = []
    for i in range(len(t_list[0])):
        if i % 20 == 0:
            print(f"Calculating curvature path done {100*i/len(t_list[0])}%")
            Zpath.append(
                curvature(
                    subloss,
                    network,
                    dataset,
                    theta=np.array([t_list[0][i], t_list[1][i]]),
                )
            )

    np.savez(
        "npfiles/curvature_plot.npz",
        X=X,
        Y=Y,
        Z=Z,
        t_list=t_list,
        Zpath=Zpath,
        allow_pickle=True,
    )

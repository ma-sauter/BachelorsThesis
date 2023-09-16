import matplotlib.pyplot as plt
import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
from Curvature_calculation import curvature_slow_but_working as curvature
from rich.progress import track
import pickle
import time

## Import dataset
with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)

## Define Network
from networks import OneNode_DB_network

network = OneNode_DB_network.network

## Define Loss functions
from losses import MeanPowerLoss2

lossname = "MeanPowerLoss2"
loss = MeanPowerLoss2.loss
subloss = MeanPowerLoss2.subloss


CALCULATE_TRAINING_AND_LOSS_SURFACE = False
CALCULATE_SCALAR_CURVATURE = True
CALCULATE_FISHER_MATRIX = False


if CALCULATE_TRAINING_AND_LOSS_SURFACE:
    # Initialize starting parameters
    1 + 1


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
                subloss, network, dataset, theta=np.array([theta1[i], theta2[j]])
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

if CALCULATE_FISHER_MATRIX:
    theta1 = np.linspace(-2, 1, 100)
    theta2 = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(theta1, theta2)
    t_list = np.load("npfiles/training.npz")["t_list"]
    l_list = np.load("npfiles/training.npz")["l_list"]
    acc = np.load("npfiles/training.npz")["acc"]

    Z11 = onp.zeros_like(X)
    Z12 = onp.zeros_like(X)
    Z22 = onp.zeros_like(X)
    for i in track(range(len(theta1))):
        for j in range(len(theta2)):
            fisher = fisher_info(
                subloss, network, dataset, theta=np.array([theta1[i], theta2[j]])
            )
            Z11[j, i] = fisher[0, 0]
            Z12[j, i] = fisher[0, 1]
            Z22[j, i] = fisher[1, 1]

    Zpath11, Zpath12, Zpath22 = [], [], []
    for i in range(len(t_list[0])):
        if i % 20 == 0:
            print(f"Calculating fisher path done {100*i/len(t_list[0])}%")
            fisher = fisher_info(
                subloss,
                network,
                dataset,
                theta=np.array([t_list[0][i], t_list[1][i]]),
            )
            Zpath11.append(fisher[0, 0])
            Zpath12.append(fisher[0, 1])
            Zpath22.append(fisher[1, 1])

    np.savez(
        "npfiles/Fisher_infos.npz",
        X=X,
        Y=Y,
        Z11=Z11,
        Z12=Z12,
        Z22=Z22,
        Zpath11=Zpath11,
        Zpath12=Zpath12,
        Zpath22=Zpath22,
        t_list=t_list,
        allow_pickle=True,
    )

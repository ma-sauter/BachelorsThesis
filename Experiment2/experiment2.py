import matplotlib.pyplot as plt
from matplotlib import cm
import jax.numpy as np
import jax
from jax import grad, jit
import numpy as onp
from fisher_calculation import fisher_info
from Curvature_calculation import curvature_slow_but_working as curvature
from NTK_calculation import NTK_trace
from rich.progress import track
import pickle
import time
import os

## Import dataset
with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)

## Define Network
from networks import OneNode_DB_network

network = OneNode_DB_network.network

## Define Loss functions
from losses import MeanPowerLoss2 as loss_functions

# Remember to also change the loss import above and the thetalims!
lossname = "MeanPowerLoss2"
loss = loss_functions.loss
subloss = loss_functions.subloss
thetalim1, thetalim2 = -4, 1


CALCULATE_TRAINING_AND_LOSS_SURFACE = True
CALCULATE_LONG_TRAINING = False
CALCULATE_SCALAR_CURVATURE = False
CALCULATE_FISHER_MATRIX = False
CALCULATE_NTK = False


if CALCULATE_TRAINING_AND_LOSS_SURFACE:
    # Traning
    #########
    n_epochs = 1000
    if lossname[:-1] == "MeanPowerLoss":
        learning_rate = 250e-3
    if lossname[:-1] == "LPNormLoss":
        learning_rate = 5e-3
    if lossname == "CrossEntropyLoss":
        learning_rate = 1e-3
    theta = np.array([0.5, 0.5])
    # Initialize starting parameters
    lossgradient = grad(loss, argnums=1)
    theta_list = [theta]
    loss_list = [loss(dataset, theta, network)]
    accuracy = []

    @jit
    def update_step(theta):
        return theta - learning_rate * lossgradient(dataset, theta, network)

    for i in track(range(n_epochs), description="Training:"):
        theta = update_step(theta)
        loss_list.append(loss(dataset, theta, network))
        theta_list.append(theta)
        wrong_guesses = 0
        N = len(dataset)
        for i in range(N):
            wrong_guesses += (
                np.round(network(dataset["inputs"][i], theta)) - dataset["targets"][i]
            ) ** 2
        accuracy.append((N - wrong_guesses) / N)

    # plt.plot(loss_list)
    # plt.show()

    # Loss Surface
    ##############
    theta1 = np.linspace(thetalim1, thetalim2, 50)
    theta2 = np.linspace(-thetalim2, -thetalim1, 50)
    X, Y = np.meshgrid(theta1, theta2)
    loss_jit = loss  # Jitting doesn't work in the nested loop below
    Z = onp.zeros_like(X)
    for i in track(range(len(theta1)), description="Loss surface calculation:"):
        for j in range(len(theta2)):
            Z[j, i] = loss_jit(
                dataset=dataset, theta=np.array([theta1[i], theta2[j]]), network=network
            )

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, Z, cmap=cm.magma)
    # theta_list = onp.transpose(theta_list)
    # ax.plot(theta_list[0], theta_list[1], loss_list, color="mediumseagreen", zorder=10)
    # plt.show()

    # Function surface
    Xfunc = onp.linspace(0, 1, 500)
    Yfunc = onp.linspace(0, 1, 500)
    Zfunc = onp.zeros(shape=(len(Xfunc), len(Yfunc)))
    t_list = onp.transpose(theta_list)
    for i in track(range(len(Xfunc)), description="Calculating function surface..."):
        x = Xfunc[i]
        for j, y in enumerate(Yfunc):
            Zfunc[j, i] = network(input=[x, y], theta=[t_list[0][-1], t_list[1][-1]])
    Xfunc, Yfunc = np.meshgrid(Xfunc, Yfunc)

    np.savez(
        f"npfiles/{lossname}_training.npz",
        l_list=loss_list,
        acc_list=accuracy,
        t_list=onp.transpose(theta_list),
        SurfX=X,
        SurfY=Y,
        SurfZ=Z,
        Xfunc=Xfunc,
        Yfunc=Yfunc,
        Zfunc=Zfunc,
    )

if CALCULATE_LONG_TRAINING:
    # Traning
    #########
    n_epochs = 1000000
    learning_rate = 50e-3
    theta = np.array([0.5, 0.5])
    # Initialize starting parameters
    lossgradient = grad(loss, argnums=1)
    theta_list = [theta]
    loss_list = [loss(dataset, theta, network)]
    curvature_list = [curvature(subloss, network, dataset, theta).item()]
    NTK_trace_list = [NTK_trace(network, dataset, theta)]
    fisher_inf = fisher_info(subloss, network, dataset, theta)
    Fisher_trace_list = [fisher_inf[0, 0] + fisher_inf[1, 1]]
    accuracy = []

    @jit
    def update_step(theta):
        return theta - learning_rate * lossgradient(dataset, theta, network)

    for i in track(range(n_epochs), description="Training:"):
        theta = update_step(theta)
        if i % 2000 == 0:
            loss_list.append(loss(dataset, theta, network))
            theta_list.append(theta)
            curvature_list.append(curvature(subloss, network, dataset, theta).item())
            NTK_trace_list.append(NTK_trace(network, dataset, theta))
            fisher_inf = fisher_info(subloss, network, dataset, theta)
            Fisher_trace_list.append(fisher_inf[0, 0] + fisher_inf[1, 1])
            wrong_guesses = 0
            N = len(dataset)
            for i in range(N):
                wrong_guesses += (
                    np.round(network(dataset["inputs"][i], theta))
                    - dataset["targets"][i]
                ) ** 2
            accuracy.append((N - wrong_guesses) / N)

    """
    n_y_points = 15
    theta1 = np.linspace(1, -6, 50)
    X, Y, Z = (
        onp.zeros(shape=(n_y_points, len(theta1))),
        onp.zeros(shape=(n_y_points, len(theta1))),
        onp.zeros(shape=(n_y_points, len(theta1))),
    )
    for i in track(range(len(theta1)), description="Long Training Curv Surface"):
        theta2 = np.linspace(-theta1[i] - 1.2, -theta1[i] + 1.2, n_y_points)
        for j in range(len(theta2)):
            X[j, i] = theta1[i]
            Y[j, i] = theta2[j]
            Z[j, i] = curvature(subloss, network, dataset, theta=[theta1[i], theta2[j]])
    """

    np.savez(
        f"npfiles/{lossname}_long_training.npz",
        t_list=onp.transpose(theta_list),
        l_list=loss_list,
        c_list=curvature_list,
        a_list=accuracy,
        NTK_trace_list=NTK_trace_list,
        Fisher_trace_list=Fisher_trace_list,
        # Xcurvsurf=X,
        # Ycurvsurf=Y,
        # Zcurvsurf=Z,
        description=f"{n_epochs} Epochs with {learning_rate} learning_rate",
    )


if CALCULATE_SCALAR_CURVATURE:
    # Scalar Curvature

    theta1 = np.linspace(thetalim1, thetalim2, 50)
    theta2 = np.linspace(-thetalim2, -thetalim1, 50)
    X, Y = np.meshgrid(theta1, theta2)
    t_list = np.load(f"npfiles/{lossname}_training.npz")["t_list"]
    l_list = np.load(f"npfiles/{lossname}_training.npz")["l_list"]

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
        f"npfiles/{lossname}curvature_plot.npz",
        X=X,
        Y=Y,
        Z=Z,
        t_list=t_list,
        Zpath=Zpath,
        allow_pickle=True,
    )

if CALCULATE_FISHER_MATRIX:
    theta1 = np.linspace(thetalim1, thetalim2, 50)
    theta2 = np.linspace(-thetalim2, -thetalim1, 50)
    X, Y = np.meshgrid(theta1, theta2)
    t_list = np.load(f"npfiles/{lossname}_training.npz")["t_list"]
    l_list = np.load(f"npfiles/{lossname}_training.npz")["l_list"]

    Z11 = onp.zeros_like(X)
    Z12 = onp.zeros_like(X)
    Z22 = onp.zeros_like(X)
    for i in track(range(len(theta1)), description="Fisher Surface..."):
        for j in range(len(theta2)):
            fisher = fisher_info(
                subloss, network, dataset, theta=np.array([theta1[i], theta2[j]])
            )
            Z11[j, i] = fisher[0, 0]
            Z12[j, i] = fisher[0, 1]
            Z22[j, i] = fisher[1, 1]

    Zpath11, Zpath12, Zpath22 = [], [], []
    for i in track(range(len(t_list[0])), description="Fisher paths..."):
        if i % 1 == 0:
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
        f"npfiles/{lossname}_fisher_infos.npz",
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

if CALCULATE_NTK:
    assert os.path.exists(f"npfiles/{lossname}_training.npz")
    data = np.load(f"npfiles/{lossname}_training.npz")
    t_list = data["t_list"]

    NTK_path = []
    for i in track(range(len(t_list[0])), description="Calculating NTK Trace..."):
        NTK_path.append(NTK_trace(network, dataset, theta=[t_list[0][i], t_list[1][i]]))

    theta1 = np.linspace(thetalim1, thetalim2, 50)
    theta2 = np.linspace(-thetalim2, -thetalim1, 50)
    X, Y = np.meshgrid(theta1, theta2)

    Z = onp.zeros_like(X)
    for i in track(range(len(theta1)), description="NTK_surface..."):
        for j in range(len(theta2)):
            Z[j, i] = NTK_trace(network, dataset, theta=[theta1[i], theta2[j]])

    np.savez(
        f"npfiles/{lossname}_ntk.npz", t_list=t_list, NTK_path=NTK_path, X=X, Y=Y, Z=Z
    )

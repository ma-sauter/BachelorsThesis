import matplotlib.pyplot as plt
import jax.numpy as np
import jax
from jax import grad
import numpy as onp
from fisher_calculation import fisher_info
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

loss = MeanPowerLoss2.loss
subloss = MeanPowerLoss2.subloss

theta = np.array([1.0, 1.0])

inputs = dataset["inputs"]
targets = dataset["targets"]

fisher_info(subloss, network, dataset, theta)
start = time.time()
print(fisher_info(subloss, network, dataset, theta))
end = time.time()

print(f"This took {end-start}")

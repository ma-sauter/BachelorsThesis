import jax.numpy as np
from jax import jit
from typing import Union


def loss(dataset: np.ndarray, theta: np.ndarray, network: callable):
    inputs = dataset["inputs"]
    targets = dataset["targets"]

    loss = 0
    for i in range(len(inputs)):
        loss += -(targets[i]) * np.log(network(inputs[i], theta))
        loss += -(1 - targets[i]) * np.log(1 - network(inputs[i], theta))
    return loss


def subloss(
    input: Union[np.ndarray, float],
    target: Union[np.ndarray, float],
    theta: np.ndarray,
    network: callable,
):
    """
    The subloss is defined for the calculation of the fisher matrix
    """
    return -(target) * np.log(network(input, theta)) - (1 - target) * np.log(
        1 - network(input, theta)
    )

import jax.numpy as np
from jax import jit
from typing import Union


def loss(dataset: np.ndarray, theta: np.ndarray, network: callable):
    """
    Computes the mean power loss with power 2

    Dataset is supposed to be an array with shape (n_data_points,2)
    """

    inputs = dataset["inputs"]
    targets = dataset["targets"]

    loss = 0
    N = inputs.shape[0]

    for i in range(N):
        loss += ((targets[i] - network(inputs[i], theta)) ** 2) ** (1 / 2)
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
    return ((target - network(input, theta)) ** 2) ** (1 / 2)

import jax.numpy as np
from jax import jit, grad


def fisher_info(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    """
    Calculation of the fisher information for parameters indexed by i and j.
    """
    inputs = dataset["inputs"]
    targets = dataset["targets"]

    subloss_gradient = grad(subloss, 2)

    fisher_matrix = np.zeros(shape=(len(theta), len(theta)))

    N_inputs = inputs.shape[0]
    for i in range(N_inputs):
        gradient = subloss_gradient(inputs[i], targets[i], theta, network)
        fisher_matrix += np.einsum("i,j->ij", gradient, gradient)

    return fisher_matrix / N_inputs

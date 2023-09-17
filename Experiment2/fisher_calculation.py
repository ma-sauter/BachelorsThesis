import jax.numpy as np
from jax import grad, debug, jit, vmap as jvmap


def fisher_info(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    """
    Calculation of the fisher information for parameters indexed by i and j.
    """
    inputs = np.array(dataset["inputs"])
    targets = np.array(dataset["targets"])

    subloss_gradient = grad(subloss, 2)

    def vmap_func(i):
        gradient = np.array(subloss_gradient(inputs[i], targets[i], theta, network))
        return np.einsum("i,j->ij", gradient, gradient)

    vmap = jvmap(vmap_func)
    fisher_list = vmap(np.arange(len(inputs)))

    return np.mean(fisher_list, axis=0)

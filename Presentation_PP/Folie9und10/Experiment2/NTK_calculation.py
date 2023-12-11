import jax.numpy as np
from jax import vmap as jvmap
from jax import jacfwd, jacrev, grad, debug, jit


def NTK_trace(network: callable, dataset: np.ndarray, theta: np.ndarray):
    inputs = np.array(dataset["inputs"])
    trace = 0
    """
    for i in range(len(inputs)):
        subnetwork = lambda x: network(inputs[i], x)
        gradient = np.array(grad(subnetwork)(theta))
        trace += np.dot(gradient, gradient)
    return trace
    """

    def vmap_func(i):
        subnetwork = lambda x: network(inputs[i], x)
        gradient = np.array(grad(subnetwork)(theta))
        return np.dot(gradient, gradient)

    vmap = jvmap(vmap_func)
    return np.sum(vmap(np.arange(len(inputs))))

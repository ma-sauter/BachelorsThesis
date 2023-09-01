import jax.numpy as np
from jax import jit


@jit
def network(input: np.ndarray, theta: np.ndarray):
    """
    Network that converst inputs to outputs depending on the parameters theta
    """
    a = 5
    return 1 / (1 + np.exp(-a * input[0] * theta[0] - a * input[1] * theta[1]))

import jax.numpy as np
from jax import jacfwd, grad
from fisher_calculation import fisher_info


def curvature(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    fisher_derivatives = jacfwd(fisher_info, argnums=3)(
        subloss, network, dataset, theta
    )

    def christoffel(i, j, k):
        symbol = 0
        i, j, k = int(i), int(j), int(k)
        for m in range(len(theta)):
            symbol += ig[m, i] * (
                fisher_derivatives[k][m, j]
                + fisher_derivatives[j][m, k]
                - fisher_derivatives[m][j, k]
            )
        return 0.5 * symbol

    # Derivatives of the christoffel symbol
    dChristoffel = [grad(christoffel, 0), grad(christoffel, 1), grad(christoffel, 2)]

    curvature = 0
    n_t = len(theta)

    for i in range(n_t):
        for j in range(n_t):
            for m in range(n_t):
                for n in range(n_t):
                    curvature += ig[i, j] * (
                        dChristoffel[m](m * 1.0, i * 1.0, j * 1.0)
                        - dChristoffel[j](m * 1.0, i * 1.0, m * 1.0)
                        + christoffel(n, i, j) * christoffel(m, m, n)
                        - christoffel(n, i, m) * christoffel(m, j, n)
                    )
    return curvature

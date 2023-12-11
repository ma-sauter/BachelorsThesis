import jax.numpy as np
from jax import vmap as jvmap
from jax import jacfwd, jacrev, grad, debug, jit
from fisher_calculation import fisher_info


def curvature1(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    fisher_derivatives = jacfwd(fisher_info, argnums=3)(
        subloss, network, dataset, theta
    )

    def christoffel(i, j, k, theta):
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
    dChristoffel = grad(christoffel, argnums=3)

    curvature = 0
    n_t = len(theta)

    for i in range(n_t):
        for j in range(n_t):
            for m in range(n_t):
                for n in range(n_t):
                    curvature += ig[i, j] * (
                        dChristoffel[m](m * 1.0, i * 1.0, j * 1.0, theta)
                        - dChristoffel[j](m * 1.0, i * 1.0, m * 1.0, theta)
                        + christoffel(n, i, j, theta) * christoffel(m, m, n, theta)
                        - christoffel(n, i, m, theta) * christoffel(m, j, n, theta)
                    )
    return curvature


def curvature2(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    hessian = jacfwd(jacfwd(fisher_info, argnums=3), argnums=3)(
        subloss, network, dataset, theta
    )

    n = len(theta)
    R = 0
    for mu in range(n):
        for v in range(n):
            for alpha in range(n):
                for beta in range(n):
                    value = (
                        ig[beta, v]
                        * ig[alpha, mu]
                        * (
                            hessian[mu][beta][alpha, v]
                            - hessian[v][beta][alpha, mu]
                            + hessian[v][alpha][beta, mu]
                            - hessian[mu][alpha][alpha, mu]
                        )
                    )
                    R += value

    return R / 2


def curvature2_vmap(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    hessian = jacfwd(jacfwd(fisher_info, argnums=3), argnums=3)(
        subloss, network, dataset, theta
    )

    n = len(theta)
    R = 0

    def vmap_func(mu, v, alpha, beta, ig, hessian):
        return (
            ig[beta, v]
            * ig[alpha, mu]
            * 0.5
            * (
                hessian[mu][beta][alpha, v]
                - hessian[v][beta][alpha, mu]
                + hessian[v][alpha][beta, mu]
                - hessian[mu][alpha][beta, v]
            )
        )

    vmap1 = jvmap(vmap_func, in_axes=(0, None, None, None, None, None))
    vmap2 = jvmap(vmap1, in_axes=(None, 0, None, None, None, None))
    vmap3 = jvmap(vmap2, in_axes=(None, None, 0, None, None, None))
    vmap = jvmap(vmap3, in_axes=(None, None, None, 0, None, None))

    pars = np.arange(len(theta))
    Rlist = vmap(pars, pars, pars, pars, ig, hessian)
    return np.sum(Rlist)


def curvature_slow_but_working(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    """
    Theta values have to be floats
    """

    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    fisher_derivative = jacfwd(fisher_info, argnums=3)

    @jit
    def christoffel(i, k, l, theta):
        symbol = 0
        g = fisher_info(subloss, network, dataset, theta)
        ig = np.linalg.inv(g)
        fisher_derivatives = np.array(
            fisher_derivative(subloss, network, dataset, theta)
        )
        for m in range(len(theta)):
            symbol += (
                0.5
                * ig[i, m]
                * (
                    fisher_derivatives[l][m, k]
                    + fisher_derivatives[k][m, l]
                    - fisher_derivatives[m][k, l]
                )
            )

        return symbol

    # Derivatives of the christoffel symbol
    dChristoffel = jit(grad(christoffel, argnums=3))

    def Riemann_tensor(alpha, beta, mu, v, theta):
        tensor = (
            dChristoffel(alpha, beta, v, theta)[mu]
            - dChristoffel(alpha, beta, mu, theta)[v]
        )
        for sigma in range(len(theta)):
            tensor += christoffel(alpha, sigma, mu, theta) * christoffel(
                sigma, beta, v, theta
            )
            tensor -= christoffel(alpha, sigma, v, theta) * christoffel(
                sigma, beta, mu, theta
            )
        return tensor

    def Ricci_tensor(alpha, beta, theta):
        tensor = 0
        for mu in range(len(theta)):
            tensor += Riemann_tensor(mu, alpha, mu, beta, theta)
        return tensor

    curvature = 0

    for mu in range(len(theta)):
        for v in range(len(theta)):
            curvature += ig[mu, v] * Ricci_tensor(mu, v, theta)
    return curvature

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import plotly.graph_objects as go

import jax.numpy as np
import jax
from jax import grad, jacfwd, debug, jit, vmap as jvmap
import numpy as onp
from rich.progress import track
import pickle
import time

TEST_BASIC_METRIC1 = True
TEST_BASIC_METRIC2 = True
TEST_SCHWARZSCHILD_METRIC = True
TEST_OFFDIAGONAL_METRIC = True

TEST_FISHER_INFORMATION = True


# Curvature with christoffels
def curvature(
    subloss: callable, network: callable, dataset: np.ndarray, theta: np.ndarray
):
    """
    Theta values have to be floats
    """

    g = fisher_info(subloss, network, dataset, theta)
    ig = np.linalg.inv(g)

    fisher_derivative = jacfwd(fisher_info, argnums=3)
    # Fisher matrix, inverse fisher matrix and Fisher derivatives work as expected
    # debug.print(
    #    "Fisher Derivatives {x}",
    #    x=fisher_derivative(subloss, network, dataset, theta)[1],
    # )
    # debug.print("Inverse fisher {x}", x=ig)

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

    # Christoffel symbols also do the right thing
    r = 0.5
    teta = np.pi / 2
    mn = 2
    """
    debug.print("Christoffel {x}", x=christoffel(mn, 0, 0, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 0, 1, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 0, 2, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 0, 3, [0.0, r, teta, 0.0]))
    debug.print("")
    debug.print("Christoffel {x}", x=christoffel(mn, 1, 0, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 1, 1, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 1, 2, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 1, 3, [0.0, r, teta, 0.0]))
    debug.print("")
    debug.print("Christoffel {x}", x=christoffel(mn, 2, 0, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 2, 1, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 2, 2, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 2, 3, [0.0, r, teta, 0.0]))
    debug.print("")
    debug.print("Christoffel {x}", x=christoffel(mn, 3, 0, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 3, 1, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 3, 2, [0.0, r, teta, 0.0]))
    debug.print("Christoffel {x}", x=christoffel(mn, 3, 3, [0.0, r, teta, 0.0]))
    """

    # Derivatives of the christoffel symbol
    dChristoffel = jit(grad(christoffel, argnums=3))

    """
    index = 2
    coordinate = 2
    debug.print("dChristoffel {x}", x=dChristoffel(index, 0, 0, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 0, 1, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 0, 2, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 0, 3, theta)[coordinate])
    debug.print("")
    debug.print("dChristoffel {x}", x=dChristoffel(index, 1, 0, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 1, 1, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 1, 2, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 1, 3, theta)[coordinate])
    debug.print("")
    debug.print("dChristoffel {x}", x=dChristoffel(index, 2, 0, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 2, 1, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 2, 2, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 2, 3, theta)[coordinate])
    debug.print("")
    debug.print("dChristoffel {x}", x=dChristoffel(index, 3, 0, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 3, 1, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 3, 2, theta)[coordinate])
    debug.print("dChristoffel {x}", x=dChristoffel(index, 3, 3, theta)[coordinate])
    debug.print("")
    """

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

    """
    r = 1.0
    teta = np.pi / 2
    x, y = 1, 0
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("")
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("")
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("")
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    debug.print("Riemann {x}", x=Riemann_tensor(x, y, 0, 0, [0.0, r, teta, 1.0]))
    """

    curvature = 0

    for mu in range(len(theta)):
        for v in range(len(theta)):
            curvature += ig[mu, v] * Ricci_tensor(mu, v, theta)
    return curvature


## My curvature calculation
def subloss():
    return None


def network():
    return None


dataset = np.array([])


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
if TEST_BASIC_METRIC1:

    def fisher_info(subloss, network, dataset, theta):
        return np.array([[np.exp(theta[1]), 0], [0, 1]])

    print("----Running tests on basic metric----")
    print("Basic Test1/6")
    assert curvature(subloss, network, dataset, theta=[0.0, 5.0]) == -0.5
    print("Basic Test2/6")
    assert curvature(subloss, network, dataset, theta=[2.0, 1.0]) == -0.5
    print("Basic Test3/6")
    assert curvature(subloss, network, dataset, theta=[5.0, 0.0]) == -0.5
    print("Basic Test4/6")
    assert curvature(subloss, network, dataset, theta=[1.0, 5.0]) == -0.5
    print("Basic Test5/6")
    assert curvature(subloss, network, dataset, theta=[29.0, 1.0]) == -0.5
    print("Basic Test6/6")
    assert curvature(subloss, network, dataset, theta=[1.0, 0.01]) == -0.5
    print("-All basic tests have been successfull-")


if TEST_BASIC_METRIC2:
    r = 1.5

    def fisher_info(subloss, network, dataset, theta):
        teta, phi = theta[0], theta[1]
        return np.array([[r**2, 0], [0, r**2 * np.sin(teta) ** 2]])

    print("----Testing basic metric 2----")
    print("Basic Test1/1")
    assert (
        np.abs(curvature(subloss, network, dataset, theta=[0.1, 5.0]) - 2 / r**2)
        < 0.01
    )
    print("Basic Test2/2")
    assert (
        np.abs(curvature(subloss, network, dataset, theta=[0.3, 5.0]) - 2 / r**2)
        < 0.01
    )
    print("-All basic tests 2 have been successfull-")


if TEST_SCHWARZSCHILD_METRIC:

    def fisher_info(subloss, network, dataset, theta):
        t, r, teta, phi = theta[0], theta[1], theta[2], theta[3]
        return np.array(
            [
                [(1 - 2 / r), 0, 0, 0],
                [0, -1 / (1 - 2 / r), 0, 0],
                [0, 0, -(r**2), 0],
                [0, 0, 0, -(r**2) * np.sin(teta) ** 2],
            ]
        )

    print("----Running tests on schwarzschild metric----")
    print("Schwarzschild Test1/3")
    assert (
        np.abs(curvature(subloss, network, dataset, theta=[1.0, 0.5, np.pi / 4, 1.0]))
        < 0.01
    )
    print("Schwarzschild Test2/3")
    assert (
        np.abs(curvature(subloss, network, dataset, theta=[1.0, 4.0, np.pi / 4, 2.0]))
        < 0.01
    )
    print("Schwarzschild Test3/3")
    assert (
        np.abs(curvature(subloss, network, dataset, theta=[1.0, 5.0, 0.2 * np.pi, 1.0]))
        < 0.01
    )
    print("-All Schwarzschild tests successfull-")


if TEST_OFFDIAGONAL_METRIC:
    print("----Starting offdiagonal metric tests----")

    def fisher_info(subloss, network, dataset, theta):
        z, r = theta[0], theta[1]
        return np.array(
            [
                [r, r],
                [r, z],
            ]
        )

    def test():
        assert (
            np.abs(
                curvature(subloss, network, dataset, [z, r])
                - (-r + z / 2) / (r**2 * (r**2 - 2 * r * z + z**2))
            )
            < 0.01
        )

    print("Running offdiagonal test 1/6")
    z, r = 1.5, 1.0
    test()
    print("Running offdiagonal test 2/6")
    z, r = 1.5, 2.0
    test()
    print("Running offdiagonal test 3/6")
    z, r = 0.5, 15.0
    test()
    print("Running offdiagonal test 4/6")
    z, r = 1.5, 0.5
    test()
    print("Running offdiagonal test 5/6")
    z, r = 0.001, -1.0
    test()
    print("Running offdiagonal test 6/6")
    z, r = -4.0, 3.0
    test()
    print("-All offdiagonal tests passed-")


if TEST_FISHER_INFORMATION:
    print("----Running test on fisher information----")

    ##################
    ## Import dataset
    with open("npfiles/dataset.npy", "rb") as file:
        dataset = pickle.load(file)

    ## Define Network
    from networks import OneNode_DB_network

    network = OneNode_DB_network.network

    ## Define Loss functions
    from losses import MeanPowerLoss2

    lossname = "MeanPowerLoss2"
    loss = MeanPowerLoss2.loss
    subloss = MeanPowerLoss2.subloss
    ################

    def analytic_fisher_info_matrix(dataset, theta, a):
        N = len(dataset["inputs"])
        inputs, targets = dataset["inputs"], dataset["targets"]
        I11, I12, I22 = 0, 0, 0
        for i in range(N):
            n_out = network(inputs[i], theta)
            I11 += (
                2
                * (targets[i] - n_out)
                * n_out**2
                * (
                    -a
                    * inputs[i, 0]
                    * np.exp(-a * theta[0] * inputs[i, 0] - a * theta[1] * inputs[i, 1])
                )
            ) ** 2
            I22 += (
                2
                * (targets[i] - n_out)
                * n_out**2
                * (
                    -a
                    * inputs[i, 1]
                    * np.exp(-a * theta[0] * inputs[i, 0] - a * theta[1] * inputs[i, 1])
                )
            ) ** 2
            I12 += (2 * (targets[i] - n_out) * n_out**2) ** 2 * (
                a**2
                * inputs[i, 1]
                * inputs[i, 0]
                * np.exp(-a * theta[0] * inputs[i, 0] - a * theta[1] * inputs[i, 1])
            )
        return np.array([[I11, I12], [I12, I22]]) / N

    from fisher_calculation import fisher_info

    def test():
        assert (
            np.abs(
                analytic_fisher_info_matrix(dataset, theta, 5)[0, 0]
                - fisher_info(subloss, network, dataset, theta)[0, 0]
            )
            < 0.01
        )

    print("Running fisher test 1/6")
    theta = [-0.5, 1.5]
    test()
    print("Running fisher test 2/6")
    theta = [-2.5, 1.5]
    test()
    print("Running fisher test 3/6")
    theta = [-0.5, 2.5]
    test()
    print("Running fisher test 4/6")
    theta = [-0.5, 0.001]
    test()
    print("Running fisher test 5/6")
    theta = [10.5, 1.5]
    test()
    print("Running fisher test 6/6")
    theta = [0.0, 0.0]
    test()
    print("-Fisher tests successfull-")

print("--------------------------------------")
print("----All tests have been successfull---")

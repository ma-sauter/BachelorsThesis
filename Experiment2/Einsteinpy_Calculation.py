import sympy
from sympy import Sum, Indexed, exp, diff, pprint
from einsteinpy.symbolic import RicciTensor, RicciScalar, metric
from einsteinpy.symbolic.predefined import AntiDeSitter
import pickle

sympy.init_printing()

with open("npfiles/dataset.npy", "rb") as file:
    dataset = pickle.load(file)
N = len(dataset["inputs"])

theta_1, theta_2, x, i = sympy.symbols("T t x i")
ell = (
    1
    / N
    * (
        Indexed("c", i)
        - 1 / (1 + exp(-5 * theta_1 * Indexed("x", i) - 5 * theta_2 * Indexed("y", i)))
    )
    ** 2
)

fisher11 = Sum(diff(ell, theta_1) * diff(ell, theta_1), (i, 1, N))
fisher12 = Sum(diff(ell, theta_1) * diff(ell, theta_2), (i, 1, N))
fisher22 = Sum(diff(ell, theta_2) * diff(ell, theta_2), (i, 1, N))

fisher_info = metric.MetricTensor(
    sympy.Array([[fisher11, fisher12], [fisher12, fisher22]]), syms=(theta_1, theta_2)
)
pprint(fisher_info, use_unicode=False)

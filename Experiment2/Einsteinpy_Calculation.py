from sympy import symbols, diag, Function, sin, cos, exp, pprint
from einsteinpy.symbolic import (
    MetricTensor,
    RicciScalar,
    RicciTensor,
    ChristoffelSymbols,
    RiemannCurvatureTensor,
)
from einsteinpy.symbolic.predefined import Schwarzschild

# Define the symbols
t, r, theta, phi, z = symbols("t r theta phi z")

# Define the function
f = Function("f")(r)

# Define the metric for a static spherical symmetric spacetime
metric = diag(
    1 - 2 / r, -1 / (1 - 2 / r), -(r**2), -(r**2) * sin(theta) ** 2
).tolist()

# Create the metric tensor
m_obj = MetricTensor(metric, (z, r, theta, phi))
# m_obj = Schwarzschild()

# Calculate Christoffel Symbols
Christ = ChristoffelSymbols.from_metric(m_obj)

# Calculate Riemannian tensor
Riem = RiemannCurvatureTensor.from_metric(m_obj)


# Calculate Ricci Tensor
Ric = RicciTensor.from_metric(m_obj)
pprint(Christ.tensor())

# Calculate the Ricci scalar
rsc = RicciScalar.from_metric(m_obj)
ricci_scalar = rsc.expr

print(ricci_scalar)

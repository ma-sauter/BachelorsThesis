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
t, r, theta, phi = symbols("t r theta phi")

# Define the function
f = Function("f")(r)

# Define the metric for a static spherical symmetric spacetime
metric = diag(exp(r), 1).tolist()

# Create the metric tensor
m_obj = MetricTensor(metric, (t, r))
m_obj = Schwarzschild()

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

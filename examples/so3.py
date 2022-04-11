import numpy as np
import matplotlib.pyplot as plt
import lorm
from disc import discrepancySO3
%matplotlib

so3 = lorm.manif.SO3()
print(so3)

# initialize M=4 points on the rotation group SO(3) and test some conversions
p = lorm.manif.ManifoldPointArrayParameterized(so3)
M = 4
p.coords = np.random.randn(M, 9)
for x in p.coords:
    q = lorm.manif.SO3.compute_quaternion_representation(x)
    print(q)
    q_proj = lorm.manif.Sphere3.compute_stereographicprojection(q)
    print(q_proj)
    print(np.linalg.norm(x)**2)
temp = p.coords
p.local_coords = p.local_coords
np.linalg.norm(temp - p.coords)

# we are going to compute an 1-design on SO(3)
N = 1
energy = discrepancySO3.plan(M, N)
energy.f(p)
cgMethod = lorm.optim.ConjugateGradientMethod()
p_new = cgMethod.run(energy, p)

# test the quadratic approximation obtained from the Gradient and Hessian
t = lorm.manif.TangentVectorArrayParameterized(p_new)
t.coords = np.random.randn(M, 9)
f, q, x = lorm.utils.eval_objective_function_with_quadratic_approximation(energy, t)
for graph in [f, q]:
    plt.plot(graph)
# the error should be O(x^3)
plt.plot((f - q) / x**3)

# print the trace-inner products of all pairs of the 1-design
xty = np.zeros([M, M])
for i, x in enumerate(p_new.coords):
    for j, y in enumerate(p_new.coords):
        xty[i, j] = np.dot(x, y)
print(xty)

# %%

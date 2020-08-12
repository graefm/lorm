import numpy as np
import pylab as pl
import lorm
from disc import discrepancyS2

sphere2 = lorm.manif.Sphere2()
print(sphere2)

# initialize M=6 random points on the sphere S^2, check their lengths
p = lorm.manif.ManifoldPointArrayParameterized(sphere2)
M = 6
p.local_coords = np.random.randn(M, 2)
for x in p.coords:
    print(np.linalg.norm(x))

# we are going to compute a 3-design on the sphere S^2
N = 3
energy = discrepancyS2.plan(M, N)

energy.f(p)
cgMethod = lorm.optim.ConjugateGradientMethod()
p_new = cgMethod.run(energy, p)

# test the quadratic approximation obtained from the Gradient and Hessian
t = lorm.manif.TangentVectorArrayParameterized(p_new)
t.coords = 0.1*np.random.randn(M, 3)
f, q, x = lorm.utils.eval_objective_function_with_quadratic_approximation(energy, t)

for y in [f, q]:
    pl.plot(y)
# the error should be O(x^3)
pl.plot((f-q)/x**3)

# print the trace-inner products of all pairs of the 3-design (octahedron)
xty = np.zeros([6, 6])
for i, x in enumerate(p_new.coords):
    for j, y in enumerate(p_new.coords):
        xty[i, j] = np.dot(x, y)
print(xty)

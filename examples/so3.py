import lorm
from lorm import discrepancySO3
import numpy as np
import pylab as pl
%matplotlib inline

so3 = lorm.manif.SO3()
print(so3._description)

p = lorm.manif.ManifoldPointArrayParameterized(so3)
M = 4
p.coords = np.random.randn(M,9)
for x in p.coords:
    q = lorm.manif.SO3.compute_quaternion_representation(x)
    print(q)
    q_proj = lorm.manif.Sphere3.compute_stereographicprojection(q)
    print(q_proj)
    print(np.linalg.norm(x)**2)
temp = p.coords
p.local_coords = p.local_coords
np.linalg.norm(temp-p.coords)

N = 1
testf = discrepancySO3.plan(M,N)


#discS2.f(p)
#discS2.grad(p).local_coords
#testf = testFunction(sphere2)
#testf = lorm.funcs.PotentialEnergyDotProductKernel(so3,lorm.cfuncs.LogKernel(),parameterized=True)
#M = 10
#p.coords = np.random.randn(M,3)
testf.f(p)
cgMethod = lorm.optim.ConjugateGradientMethod()
p_new = cgMethod.run(testf,p)
#p_new = p
t = lorm.manif.TangentVectorArrayParameterized(p_new)
t.coords = 0.1*np.random.randn(M,9)
f, q, x = lorm.utils.eval_objective_function_with_quadratic_approximation(testf,t)
for y in [f,q]:
        pl.plot(y)
pl.plot((f-q)/x**3)
xty = np.zeros([M,M])
for i,x in enumerate(p_new.coords):
    for j,y in enumerate(p_new.coords):
        xty[i,j] = np.dot(x,y)
print(xty)

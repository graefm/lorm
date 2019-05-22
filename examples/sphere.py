import lorm
from lorm import discrepancyS2
import numpy as np
import pylab as pl
%matplotlib inline

sphere2 = lorm.manif.Sphere2()
print(sphere2._description)

p = lorm.manif.ManifoldPointArrayParameterized(sphere2)
M = 6
p.local_coords = np.random.randn(M,2)
for x in p.coords:
    print(np.linalg.norm(x))

N = 3
testf = discrepancyS2.plan(M,N)


#discS2.f(p)
#discS2.grad(p).local_coords
#testf = testFunction(sphere2)
#testf = lorm.funcs.PotentialEnergyDotProductKernel(sphere2,lorm.cfuncs.LogKernel(),parameterized=True)
#M = 10
#p.coords = np.random.randn(M,3)
testf.f(p)
cgMethod = lorm.optim.ConjugateGradientMethod()
p_new = cgMethod.run(testf,p)
#p_new = p
t = lorm.manif.TangentVectorArrayParameterized(p_new)
t.coords = 0.1*np.random.randn(M,3)
f, q, x = lorm.utils.eval_objective_function_with_quadratic_approximation(testf,t)

for y in [f,q]:
        pl.plot(y)
pl.plot((f-q)/x**3)
xty = np.zeros([6,6])
for i,x in enumerate(p_new.coords):
    for j,y in enumerate(p_new.coords):
        xty[i,j] = np.dot(x,y)
print(xty)

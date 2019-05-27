%load_ext autoreload
%autoreload 2
import numpy as np
import lorm
import energy_stippling_s2
import energy_curvling_s2
import matplotlib.pyplot as pl
%matplotlib inline
from lorm import discrepancyS2

#img = pl.imread('pics/face.png')[:,:,0]
#img = 256-pl.imread('pics/eyeofthetiger_original.jpg')
#img = 256-pl.imread('pics/world32k_grey.jpg')
#img = 1-pl.imread('pics/sphere.png')[:,:,0]
img = 1-pl.imread('pics/world_black_white.png')
pl.imshow(img.transpose())
img.shape
Mx, My = img.shape
N = 256
plan = lorm.nfsft.plan(Mx*My,N)

x = np.linspace(0,np.pi,Mx,endpoint=False)
y = np.linspace(0,2*np.pi,My,endpoint=False)
xv, yv = np.meshgrid(x, y)
xy = np.zeros([Mx*My,2])
for i in range(Mx*My):
    xy[i,0] = xv.ravel()[i]
    xy[i,1] = yv.ravel()[i]
plan.set_local_coords(xy)

f = img.transpose()*np.sin(xy[:,0]).reshape([My,Mx])
f_hat = plan.compute_Ymatrix_adjoint_multiplication(f,N)
f = img.transpose()*np.sin(xy[:,0]).reshape([My,Mx])/f_hat[0,0]
f_hat = plan.compute_Ymatrix_adjoint_multiplication(f,N)
f_hat[0,0]

#np.savetxt("earth.fhat",f_hat._fhat_array)
f_hat = lorm.nfsft.SphericalFourierCoefficients(256)
temp = np.loadtxt('earth.fhat',dtype=complex)
f_hat._fhat_array[:] = temp
f_hat[0,0]

s2 = lorm.manif.Sphere2()
points = lorm.manif.ManifoldPointArrayParameterized(s2)

points.coords = np.loadtxt('line.txt')
m = 10000
local_coords = np.zeros([m,2])
for i in range(m):
    local_coords[i,0] = np.pi/10+0.1*np.sin(2*np.pi*(i/m))#+0.001*np.random.randn(1)#np.arccos(((i))/(m+1))
    local_coords[i,1] = 2*np.pi*(i+0.2)/(m)#np.pi*(2*(i+1)-(m+1))/10/(1+np.sqrt(5))
points.local_coords = local_coords
#points.local_coords = np.random.randn(m,2)

#points.coords=temp.coords
#np.savetxt('line.txt',points.coords)
#temp = points.coords
#points.coords = temp
#temp = np.array(points.coords)
#vectors = lorm.manif.TangentVectorArrayParameterized(points)
N = 250
#m=50
#points.coords = np.random.randn(m,3)
#alpha = 0.000001 #N=20 M=200 hemisphere
#alpha = 0.0000001 #N=40 M=800 hemisphere
#alpha = 0.00000001 #N=60 M=1800 hemisphere
alpha =  0.000000001
m = len(points.coords)
m
#alpha = 0
#disc=energy_stippling_s2.plan(m,N)
disc = energy_curvling_s2.plan(m,N,alpha,10)
for n in range(1,N+1):
    disc._lambda_hat[n,:,:] = 1./((2.*n-1)*(2*n+1)*(2*n+3))#*(2*n+5))#*(2*n+7))
#integrate(legendre_p_n(x),x,0,1)
#int_pn = [1,1/2,0,-1/8,0,1/16,0,-5/128,0,7/256,0,-21/1024,0,33/2048,0,-429/32768,0,715/65536,0,-2431/262144,0]
# gauss
#int_pn = [0.443113, 0.318113, 0.151197, 0.0304073, -0.0166321, -0.0166454, -0.00533316, 0.000670248, 0.00125445, 0.000391626, -0.0000510922, -0.0000747994, -0.0000168335, 4.65764*10**-6, 3.35405*10**-6, 3.4194*10**-7]
#int_pn = [0.157709, 0.13413, 0.0970165, 0.059671, 0.0312041, 0.013871, 0.00524029, 0.00168208, 0.000458638, 0.000106198, 0.0000208773, 3.48369*10**-6, 4.93297*10**-7, 5.92636*10**-8]
#disc._mu_hat._fhat_array[:] = 0
for n in range(N):
    disc._mu_hat[n,:] = f_hat[n,:]
#for n in range(np.min([N+1,len(int_pn)])):
#    disc._mu_hat[n,0,0] = int_pn[n]*np.sqrt(2*n+1)/int_pn[0]
#disc._weights =np.sqrt(4*np.pi)*np.real(disc._mu_hat[0,0,0])* np.ones([int(m),1],dtype=float) / int(m)
#disc._mu_hat[0,0,0] = 1
disc._mu_hat[0,0]
disc.f(points)
grad=disc.grad(points)
#grad.coords = vectors.coords
#grad=vectors
#points.coords
#grad.coords


grad.coords *= 10
#[f, q, s] =lorm.utils.eval_objective_function_with_quadratic_approximation(disc,grad)
#pl.plot(s,f,s,q)
method = lorm.optim.SteepestDescentMethod(max_iter=10)
method = lorm.optim.ConjugateGradientMethod(max_iter=20)
for i in range(2):
    points = method.run(disc,points)
    np.savetxt('line.txt',points.coords)
    #lengths=disc._eval_lengths(points.local_coords)
    #np.sum(lengths)
    #pl.plot(lengths)

pl.plot(disc._eval_lengths(points.local_coords))

import scipy as sp
from scipy import special
lengths=disc._eval_lengths(points.local_coords)
pl.plot(sp.special.sph_harm(0,1,points.local_coords[:,1],points.local_coords[:,0]))
err = disc._eval_error_vector(points.local_coords)*disc._lambda_hat
err_harm = np.zeros(N+1)
for i in range(N):
    err_harm[i] = np.sum(np.abs(err[i,-i:i]))
pl.plot(err_harm)


temp = points.coords
dist = np.zeros([m,m])
for i,x in enumerate(temp):
    for j,y in enumerate(temp):
        dist[i,j] = np.linalg.norm(x-y)
dist[18,43:]
temp1 = np.zeros([43-18,3])
for i in range(43-18):
    temp1[i,:] = points.coords[43-i-1,:]
temp[18:43,:]
temp[18:43,:]=temp1
np.savetxt('line.txt',temp)
points.coords=temp
b=np.log(1.5)/np.log(10)
1/b

10**0.177
(0.1**-b)
(0.01**-b)
1.5*(0.0000001**-b)
1.5**(1/b)
10*7**(-1/b)
-1/b*1/2
160**(-3)
2**(1.5)
0.5**(-1/b)
12 * 0.15

points.coords = temp.coords
temp = lorm.manif.ManifoldPointArray(s2)
temp.coords = points.coords

tempc = temp.coords
m2 = len(tempc)
temp.coords = np.zeros([2*m2,3])
for i in range(m2):
    temp.coords[2*i,:] = tempc[i,:]
for i in range(m2-1):
    temp.coords[2*i+1,:] = (tempc[i,:]+tempc[i+1,:])/2
temp.coords[2*m2-1,:] = (tempc[0,:]+tempc[m2-1,:])/2

m2 = len(temp.coords)
m2

%load_ext autoreload
%autoreload 2
import numpy as np
import lorm
from disc import energy_curveling_3d
from disc import energy_stippling_3d
#from disc import energy_stippling_2d
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

e3 = lorm.manif.EuclideanSpace(3)
points = lorm.manif.ManifoldPointArray(e3)

img_fhat = np.ones([10,10,10])
N_half = 5

m = 100
coords = np.zeros([m,3])
for i in range(m):
    coords[i,0] = 0.3*np.cos(2*np.pi*i/m)
    coords[i,1] = 0.3*np.sin(2*np.pi*i/m)
    coords[i,2] = 0.3*np.sin(6*np.pi*i/m)
#points.coords = temp.coords
points.coords = coords
m = len(points.coords)

#alpha = 0.0000000001
alpha = 0.005
N = 2*6
#m = 600
m = len(points.coords)
disc=energy_curveling_3d.plan(m,N,alpha,2)
#disc=energy_stippling_3d.plan(m,N)
#for i in range(N):
#    for j in range(N):
#        disc._mu_hat[i,j] = np.exp(-((i-N/2)**2+(j-N/2)**2))
#disc._mu_hat
#np.max(np.real(img_fhat))
#N_half
#img_fhat.shape
#img_fhat[N_half,N_half]
#disc._mu_hat[int(N/2)-N_half:int(N/2)+N_half,int(N/2)-N_half:int(N/2)+N_half] = img_fhat/img_fhat[N_half,N_half]
#disc._mu_hat = np.zeros([N,N])
#disc._mu_hat[N_half,N_half]=1
#points.coords = np.random.rand(m,2)-0.5
#m= len(points.coords)
disc.f(points)
m
method = lorm.optim.SteepestDescentMethod(max_iter=50)
method = lorm.optim.ConjugateGradientMethod(max_iter=150)
for i in range(1):
    points = method.run(disc,points)
x = np.zeros([m,3])
x[:] = points.coords
fig = pl.figure()
ax = fig.gca(projection='3d')
#ax.plot(x[:,1], x[:,0], x[:,2])
#ax.plot(x[:,1], x[:,2], x[:,0])
ax.plot(x[:,0], x[:,1], x[:,2])
fig






x[:,1]*=-1
#points.coords=x
v=disc.grad(points)
v=v.coords
np.linalg.norm(v)
plot_points_vecs(x,dots=True)
plot_points_vecs(x,dots=False)
pl.plot(points.coords[:,0],-points.coords[:,1],'o',markersize=0.1)
#np.savetxt('eyeofthetiger.coords',x)
#np.savetxt('trui.coords',x)
m
#disc2=energy_curvling_2d.plan(m,N,alpha,6)
#points.coords = coords0
#points = method.run(disc,points)
x=points.coords
v=disc.grad(points)
v.coords *= .10
f,q, s = lorm.utils.eval_objective_function_with_quadratic_approximation(disc,v)
pl.plot(s,f,s,q)

v=v.coords
np.linalg.norm(v)
plot_points_vecs(x,0*v)

points.coords = temp.coords
temp = lorm.manif.ManifoldPointArray(e2)
temp.coords = points.coords

tempc = temp.coords
m2 = len(tempc)
temp.coords = np.zeros([2*m2,2])
for i in range(m2):
    temp.coords[2*i,:] = tempc[i,:]
for i in range(m2-1):
    temp.coords[2*i+1,:] = (tempc[i,:]+tempc[i+1,:])/2
temp.coords[2*m2-1,:] = (tempc[0,:]+tempc[m2-1,:])/2

m2 = len(temp.coords)
m2

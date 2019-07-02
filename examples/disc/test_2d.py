%load_ext autoreload
%autoreload 2
import numpy as np
import lorm
import nfft
from disc import energy_curveling_2d
from disc import energy_stippling_2d
import matplotlib.pyplot as pl
%matplotlib inline

def plot_points_vecs(points,vecs=None, dots=True):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    m = len(points)
    for k in [-1,0,1]:
        for l in [-1,0,1]:
            x = points + np.array([k,l])
            if vecs is not None:
                v = vecs
                vn = np.linalg.norm(v)
                for i in range(m):
                    ax.arrow(x[i,0],x[i,1],v[i,0]/vn,v[i,1]/vn)
                #x=points.coords
            if dots is True:
                ax.plot(x[:,0], x[:,1],'ro', markersize=3)

            lin = pl.Line2D(x[:,0],x[:,1])
            ax.add_line(lin)
            lin = pl.Line2D([x[m-1,0],x[0,0]],[x[m-1,1],x[0,1]])
            ax.add_line(lin)
    ax.axis(1.*np.array([-0.5,0.5,-0.5,0.5]))
    ax.axes.set_aspect(1)
    fig


e2 = lorm.manif.EuclideanSpace(2)
points = lorm.manif.ManifoldPointArray(e2)

#temp = points.coords
#img = np.ones([256,256])
def compute_img_fhat(N_half):
    #img = 1-pl.imread('gaussian2d.png')
    #pl.imshow(img)
    #img = 1-pl.imread('trui.png')
    #img = np.ones([512,512],dtype=float)
    img = 256-pl.imread('data/eyeofthetiger.jpg')
    N = int(img.shape[0]/2)
    fft_img = np.fft.fftshift(np.fft.ifft2(img))[N-N_half:N+N_half,N-N_half:N+N_half]
    img_scaled=np.real(np.fft.fft2(np.fft.fftshift(fft_img)))
    #pl.imshow(1-img_scaled)
    Mx = 2*N_half
    My = 2*N_half
    x = np.linspace(-0.5,0.5,Mx,endpoint=False)
    y = np.linspace(-0.5,0.5,My,endpoint=False)
    xv, yv = np.meshgrid(x, y)
    xy = np.zeros([Mx*My,2])
    for i in range(Mx*My):
        xy[i,0] = xv.ravel()[i]
        xy[i,1] = yv.ravel()[i]
    plan = nfft.nfft.NFFT2D(Mx*My,2*N_half,2*N_half)
    plan.x = xy
    plan.precompute_x()
    plan.f = img_scaled/Mx/My
    plan.adjoint()
    img_fhat = np.zeros([2*N_half,2*N_half],dtype=np.complex)
    plan.f[:] *= 0
    img_fhat[:] = plan.f_hat
    #plan.trafo()
    #pl.imshow(np.real(plan.f.reshape([My,Mx])))
    return img_fhat

N_half = 85
img_fhat=compute_img_fhat(N_half)
pl.imshow(np.abs(img_fhat))

#disc=energy_curvling_2
#points.coords = generate_grid_points(10)
m = 100000#20000
coords = np.zeros([m,2])
for i in range(m):
    coords[i,0] = 0.3*np.cos(2*np.pi*i/m)
    coords[i,1] = 0.3*np.sin(2*np.pi*i/m)
#points.coords = temp.coords
points.coords = coords
m = len(points.coords)
m
pl.plot(points.coords[:,0],points.coords[:,1])
#temp.coords = points.coords
#points.coords = temp.coords
alpha =  0.5*0.0000001
N = 600
#m = 600
m = len(points.coords)
disc=energy_curveling_2d.plan(m,N,alpha,2)
#disc=energy_stippling_2d.plan(m,N)
#for i in range(N):
    #    for j in range(N):
#        disc._mu_hat[i,j] = np.exp(-((i-N/2)**2+(j-N/2)**2))
#disc._mu_hat
#np.max(np.real(img_fhat))
#N_half
#img_fhat.shape
#img_fhat[N_half,N_half]
disc._mu_hat[int(N/2)-N_half:int(N/2)+N_half,int(N/2)-N_half:int(N/2)+N_half] = img_fhat/img_fhat[N_half,N_half]
#disc._mu_hat = np.zeros([N,N])
#disc._mu_hat[N_half,N_half]=1
#points.coords = np.random.rand(m,2)-0.5
#m= len(points.coords)
disc.f(points)
m
method = lorm.optim.SteepestDescentMethod(max_iter=20)
method = lorm.optim.ConjugateGradientMethod(max_iter=100)
for i in range(1):
    points = method.run(disc,points)

x = np.zeros([m,2])
x[:] = points.coords
x[:,1]*=-1
#points.coords=x
v=disc.grad(points)
v=v.coords
np.linalg.norm(v)
plot_points_vecs(x,dots=True)
plot_points_vecs(x,dots=False)
pl.plot(points.coords[:,0],-points.coords[:,1],'o',markersize=0.2)
np.savetxt('eyeofthetiger.coords',x)
#np.savetxt('trui.coords',x)
m
#disc2=energy_curvling_2d.plan(m,N,alpha,6)
#points.coords = coords0
#points = method.run(disc,points)
x=points.coords
v=disc.grad(points)
v.coords *= 10
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

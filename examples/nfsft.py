import lorm
import numpy as np
import pylab as pl
%matplotlib inline


Mp, Mt = 160,81
N = 3
nfsft = lorm.nfsft.plan(Mt*Mp,N)

p = np.linspace(0,2*np.pi,Mp,endpoint=False)
t = np.linspace(0.00001,np.pi-0.00001,Mt,endpoint=True)
pv, tv = np.meshgrid(p, t)
local_coords = np.zeros([Mp*Mt,2])
for i in range(Mp*Mt):
    local_coords[i,0] = tv.ravel()[i]
    local_coords[i,1] = pv.ravel()[i]
nfsft.set_local_coords(local_coords)

fhat = lorm.nfsft.SphericalFourierCoefficients(N)
plk_vals = [[1],[-1,0,-1],[3,0,-0.5,0,3],[-15,0,1.5,0,1.5,0,-15]]
for k in range(0,N+1):
    for l in range(-k,k+1):
        fhat[k,l]=(-1)**k*np.sqrt((2*k+1)*(np.math.factorial(k-np.abs(l)))/(np.math.factorial(k+np.abs(l))))*plk_vals[k][l+k]

for k in range(N+1):
    print(fhat[k,:])
f=nfsft.compute_Ymatrix_multiplication(fhat)
f=f.reshape(Mt,Mp)
pl.imshow(np.real(f))

gradf = nfsft.compute_gradYmatrix_multiplication(fhat)
dtheta = gradf[:,0].reshape(Mt,Mp)
dphi = gradf[:,1].reshape(Mt,Mp)
pl.imshow(np.real(dtheta))
pl.imshow(np.real(dphi))
pl.imshow(np.sqrt((np.real(dtheta)**2+np.real(dphi)**2)))

hessf = nfsft.compute_hessYmatrix_multiplication(fhat)
dtheta_theta = hessf[:,0].reshape(Mt,Mp)
dtheta_phi = hessf[:,1].reshape(Mt,Mp)
dphi_phi = hessf[:,2].reshape(Mt,Mp)

(hessf[:,1] * hessf[:,1]).shape

pl.imshow(np.real(dtheta_theta))
pl.imshow(np.real(dtheta_phi))
pl.imshow(np.real(dphi_phi))
gradf
fhat = nfsft.compute_gradYmatrix_adjoint_multiplication(gradf)
for n in range(N+1):
    print(fhat[n,:])

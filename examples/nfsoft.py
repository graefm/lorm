import lorm
from lorm import cnfsoft
import numpy as np
import pylab as pl
%matplotlib inline

Mp1, Mt, Mp2 = 20, 21, 20
N = 3
nfsoft = lorm.nfsoft.plan(Mp1*Mt*Mp2, N)

p1 = 2*np.pi*np.linspace(-0.5,0.5,Mp1,endpoint=False)
t =  2*np.pi*np.linspace(0+0.00001,0.5-0.00001,Mt,endpoint=True)
p2 = 2*np.pi*np.linspace(-0.5,0.5,Mp2,endpoint=False)
tv, p1v, p2v = np.meshgrid(t, p1, p2)
p1tp2 = np.zeros([Mp1*Mt*Mp2,3])
for i in range(Mp1*Mt*Mp2):
    p1tp2[i,0] = p1v.ravel()[i]
    p1tp2[i,1] = tv.ravel()[i]
    p1tp2[i,2] = p2v.ravel()[i]
nfsoft.set_local_coords(p1tp2)

fhat = lorm.nfsoft.SO3FourierCoefficients(N)
fhat[1,1,-1] = 1

f = nfsoft.compute_Ymatrix_multiplication(fhat)
fhat = nfsoft._get_fhat(N)
#for n in range(N+1):
#    print(fhat[n,:,:])

p1tp2 = p1tp2.ravel().reshape(Mp1,Mt,Mp2,3)

f = f.reshape(Mp1,Mt,Mp2)
pl.imshow(np.real(f[:,5,:]),aspect=Mp2/Mp1)
pl.plot(np.real(f[np.int(Mp1/2),:,np.int(Mp2/2)]))
gradf = nfsoft.compute_gradYmatrix_multiplication(fhat)
dphi1 = gradf[:,0]
dphi1 = dphi1.reshape(Mp1,Mt,Mp2)
pl.imshow(np.real(dphi1[:,5,:]),aspect=Mp2/Mp1)
dphi2 = gradf[:,2]
dphi2 = dphi2.reshape(Mp1,Mt,Mp2)
pl.imshow(np.real(dphi2[:,5,:]),aspect=Mp2/Mp1)
dtheta = gradf[:,1]
dtheta = dtheta.reshape(Mp1,Mt,Mp2)
pl.plot(np.real(dtheta[np.int(Mp1/2),:,np.int(Mp2/2)]))
gradf
fhat = nfsoft.compute_gradYmatrix_adjoint_multiplication(gradf)
for n in range(N+1):
    print(np.imag(fhat[n,:,:]))

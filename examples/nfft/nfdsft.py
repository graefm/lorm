import lorm
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
%matplotlib inline

from lorm.nfdsft import *

N=5
x = np.pi*(np.random.rand((2*N+2)**4,4))
x[:,(1,3)] *= 2
x[:,(1,3)] -= np.pi
f_hat = np.random.rand((N+1)**4)+1j*np.random.rand((N+1)**4)
f_hat /= np.linalg.norm(f_hat,1)
nfdsft = plan(len(x),N)
nfdsft.set_local_coords(x)
#Y = nfssft.compute_Ymatrix()#Y_matrix(N,x)
#Yf_hat = np.dot(Y,f_hat)
Yf_hat = nfdsft.compute_Ymatrix_multiplication_direct(f_hat)# np.dot(Y,f_hat)
Yf_hat_new = nfdsft.compute_Ymatrix_multiplication(f_hat)
np.linalg.norm(Yf_hat - Yf_hat_new,np.inf)



f = np.random.rand((2*N+2)**4)+1j*np.random.rand((2*N+2)**4)
f /= np.linalg.norm(f,1)
#Y_ad = Y.conj().transpose()
#Y_ad_f = np.dot(Y_ad,f)
Y_ad_f = nfdsft.compute_Ymatrix_adjoint_multiplication_direct(f)
Y_ad_f_new = nfdsft.compute_Ymatrix_adjoint_multiplication(f)
np.linalg.norm(Y_ad_f-Y_ad_f_new,np.inf)


Mt1,Mp1,Mt2,Mp2 = 15,30,15,30
M = Mt1*Mp1*Mt2*Mp2
theta1 = np.linspace(0,np.pi,Mt1)
phi1 = np.linspace(0,2*np.pi,Mp1,endpoint=False)
theta2 = np.linspace(0,np.pi,Mt2)
phi2 = np.linspace(0,2*np.pi,Mp2,endpoint=False)
p1v, t1v, t2v, p2v = np.meshgrid(phi1, theta1, theta2, phi2)
x = np.zeros([M,4])
for i in range(M):
    x[i,0] = t1v.ravel()[i]
    x[i,1] = p1v.ravel()[i]
    x[i,2] = t2v.ravel()[i]
    x[i,3] = p2v.ravel()[i]



#N=2
#M = 2*N+2
#x = np.random.rand(M**4,4)
#f_hat = np.random.rand((N+1)**4)
#f_hat = np.zeros((N+1)**4)
#index(N, 0,0, 0,1)
#f_hat[index(N, 0,0, 0,1)]=1

N = 2
fhat = np.zeros((N+1)**4)
fhat[nfdsft.linearized_index(N, -1,2, 1,1)] = 1
nfdsft = plan(M,N)
nfdsft.set_local_coords(x)
f=nfdsft.compute_Ymatrix_multiplication_direct(fhat)

f=f.reshape(Mt1,Mp1,Mt2,Mp2)
pl.imshow(np.real(f[:,:,int(Mt2/2),int(Mp2/2)]),aspect=Mp1/Mt1/2)
pl.imshow(np.real(f[int(Mt1/2),int(Mp2/2),:,:]),aspect=Mp2/Mt2/2)

N = 5
k = 3
L2E = leg_hat_to_exp_hat_transition_matrix(k,N)
L2E[:,2]
x = np.linspace(-1,1,10)
N_exp = 100
xe = np.linspace(0,N_exp-1,N_exp)/N_exp*(2*np.pi)
pl.plot(xe,sp.special.lpmv(1,1,np.cos(xe)))

from lorm import cnfft
#import sys
M = 40
N_nfft = L2E.shape[0]
x = np.linspace(0,1/2,M)
#x = np.random.rand(M)/2
plan = cnfft.plan(M,[N_nfft],[2*N_nfft],6)
plan.x = x
plan.precompute_x()
leg_hat = np.random.randn(L2E.shape[1])#np.array([0,1,0,0,0,1,0,0,0,0])
plan.f_hat = np.dot(L2E,leg_hat)
#plan.f_hat
L = leg_matrix(k,N,np.cos(2*np.pi*x))
#L
plan.trafo()
f_nfft = plan.f
f_nfft.shape
np.linalg.norm(np.dot(L,leg_hat)-np.real(f_nfft))

for y in [np.dot(L,leg_hat),np.real(f_nfft)]:
    pl.plot(y)

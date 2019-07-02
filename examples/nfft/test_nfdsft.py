%load_ext autoreload
%autoreload 2
import nfft
import lorm
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
%matplotlib inline

import disc
N = 4
M = 48 #int(1.3*(N+1)**4/2/4)

#3-design construction
#octa = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
#points.coords = np.random.randn(2*M,3)
#for i in range(6):
#    for j in range(3):
#        points.coords[6*i+2*j,:] = octa[i]
#for i in range(6):
#    for j in range(3):
#        points.coords[6*i+2*j+1,:] = octa[j]
#points.coords.reshape(18,6)
#points.coords = points.coords


(np.loadtxt("code48.txt"))
s2.compute_inverse_parameterization(np.array([[1,0,0]]))

points.coords = np.loadtxt("code48.txt").reshape(2*48,3)
points.local_coords
#48 code construction
#octa = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
#points.coords = np.random.randn(2*M,3)
#for i in range(6):
#    for j in range(3):
#        points.coords[6*i+2*j,:] = octa[i]
#for i in range(6):
#    for j in range(3):
#        points.coords[6*i+2*j+1,:] = octa[j]
#points.coords.reshape(18,6)
#points.coords = points.coords


energy = disc.energy_stippling_g24.plan(M,N,m=2,sigma=1)
energy._N
energy._lambda_hat[4,2,:,:]
s2 = lorm.manif.Sphere2()
points = lorm.manif.ManifoldPointArrayParameterized(s2)
points.coords = np.random.randn(2*M,3)
#points.coords
#points.coords[2,:] = points.coords[0,:]
#points.coords[3,:] = -points.coords[1,:]
energy.f(points)
grad = energy.grad(points)
f,q,s = lorm.utils.eval_objective_function_with_quadratic_approximation(energy,grad)

pl.plot(s,f,s,q)
#np.abs(energy._eval_error_vector(points.local_coords).array*energy._lambda_hat.array)

method = lorm.optim.ConjugateGradientMethod(max_iter=80)
method.run(energy,points)
count = 0
points_min = lorm.manif.ManifoldPointArrayParameterized(s2)
points_min.coords = points.coords
fmin = energy.f(points_min)
    for i in range(20):
        points.coords = np.random.randn(2*M,3)
        points=method.run(energy,points)
        f = energy.f(points)
        if f < fmin:
            points_min.coords = points.coords
            fmin = energy.f(points_min)
        if energy.f(points) < 1e-4:
            count += 1
count
i
fmin
points.coords=points_min.coords
#points.coords.reshape(M,6)
dist = np.zeros([M,M])
for i in range(M):
    for j in range(M):
        dist[i,j] = 1-np.dot(points.coords[2*i,:],points.coords[2*j,:])*np.dot(points.coords[2*i+1,:],points.coords[2*j+1,:])
dist
#np.savetxt('g24_3_design.txt',points.coords)
grad = energy.grad(points)
grad.coords *= .10
grad.perform_geodesic_step()
energy.f(grad.base_point_array)
f,q,s = lorm.utils.eval_objective_function_with_quadratic_approximation(energy,grad)
pl.plot(s,f,s,q)


class test(lorm.funcs.ManifoldObjectiveFunction):
    def __init__(self,N):
        self._plan = nfft.nfdsft.plan(1,N)
        self._fhat = f_hat = nfft.nfdsft.DoubleSphericalFourierCoefficients(N)
        self._fhat[0,0,0,0] = 1

        def f(point_array_coords):
            self._plan.set_local_coords(point_array_coords.reshape(1,4))
            return np.real(self._plan.compute_Ymatrix_multiplication(self._fhat))

        def grad(point_array_coords):
            self._plan.set_local_coords(point_array_coords.reshape(1,4))
            return np.real(self._plan.compute_gradYmatrix_multiplication(self._fhat).reshape(2,2))

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-7
            return norm*(self._grad(base_point_array_coords + h*tangent_array_coords/norm) - self._grad(base_point_array_coords))/h

        lorm.funcs.ManifoldObjectiveFunction.__init__(self,lorm.manif.Sphere2(),f,grad=grad,hess_mult=hess_mult, parameterized=True)

tensorHarm = test(3)
16*np.pi**2*(tensorHarm.f(points))**2
points.local_coords
v = lorm.manif.TangentVectorArrayParameterized(points)
v.local_coords +=1
v.perform_geodesic_step()
#v.base_point_array.coords
tensorHarm._fhat[3,3,-2,1] = 30

tensorHarm.f(points)
grad = tensorHarm.grad(points)
grad.coords
v = lorm.manif.TangentVectorArrayParameterized(points)
v.coords = tensorHarm.grad(points).coords
#v.perform_geodesic_step()
v.coords *= 10
f,q,s = lorm.utils.eval_objective_function_with_quadratic_approximation(tensorHarm,v)
pl.plot(s,(f-q)/s**3)
pl.plot(s,f,s,q)
method = lorm.optim.ConjugateGradientMethod(max_iter=6)
points = method.run(tensorHarm,points)

N=4
x = np.pi*(np.random.rand((2*N+2)**4,4))
x[:,(1,3)] *= 2
x[:,(1,3)] -= np.pi
f_hat = nfft.nfdsft.DoubleSphericalFourierCoefficients(N)
f_hat.array[:] = np.random.rand((N+1)**4)+1j*np.random.rand((N+1)**4)
f_hat.array[:] /= np.linalg.norm(f_hat.array[:],1)
nfdsft_plan = nfft.nfdsft.plan(len(x),N)
nfdsft_plan.set_local_coords(x)
#Y = nfssft.compute_Ymatrix()#Y_matrix(N,x)
#Yf_hat = np.dot(Y,f_hat)
Yf_hat = nfdsft_plan.compute_Ymatrix_multiplication_direct(f_hat)# np.dot(Y,f_hat)
Yf_hat_new = nfdsft_plan.compute_Ymatrix_multiplication(f_hat)
np.linalg.norm(Yf_hat - Yf_hat_new,np.inf)



f = np.random.rand((2*N+2)**4)+1j*np.random.rand((2*N+2)**4)
f /= np.linalg.norm(f,1)
#Y_ad = Y.conj().transpose()
#Y_ad_f = np.dot(Y_ad,f)
Y_ad_f = nfdsft_plan.compute_Ymatrix_adjoint_multiplication_direct(f,N)
Y_ad_f_new = nfdsft_plan.compute_Ymatrix_adjoint_multiplication(f,N)
np.linalg.norm(Y_ad_f.array-Y_ad_f_new.array,np.inf)

Mt1,Mp1,Mt2,Mp2 = 15,30,15,30
M = Mt1*Mp1*Mt2*Mp2
theta1 = np.linspace(0+0.001,np.pi-0.001,Mt1)
phi1 = np.linspace(0,2*np.pi,Mp1,endpoint=False)
theta2 = np.linspace(0+0.001,np.pi-0.001,Mt2)
phi2 = np.linspace(0,2*np.pi,Mp2,endpoint=False)
p1v, t1v, t2v, p2v = np.meshgrid(phi1, theta1, theta2, phi2)
x = np.zeros([M,4])
for i in range(M):
    x[i,0] = t1v.ravel()[i]
    x[i,1] = p1v.ravel()[i]
    x[i,2] = t2v.ravel()[i]
    x[i,3] = p2v.ravel()[i]

N = 3
nfdsft_plan = nfft.nfdsft.plan(M,N)
nfdsft_plan.set_local_coords(x)
f_hat = nfft.nfdsft.DoubleSphericalFourierCoefficients(N-1)
f_hat[2,2,0,0] = 1
#f_hat.array[:] = np.zeros((N+1)**4,dtype=np.complex)
#f_hat.array[nfft.nfdsft.linearized_index(N,-1,3,0,0)] = 1
f = nfdsft_plan.compute_Ymatrix_multiplication(f_hat)

pl.imshow(np.real(f.reshape((Mt1,Mp1,Mt2,Mp2))[:,:,int(Mt2/2),int(Mp2/2)]))
pl.imshow(np.real(f.reshape((Mt1,Mp1,Mt2,Mp2))[int(Mt1/2),int(Mp2/2),:,:]))
fgrad = nfdsft_plan.compute_gradYmatrix_multiplication(f_hat)
pl.imshow(np.real(fgrad[:,0].reshape((Mt1,Mp1,Mt2,Mp2))[:,:,int(Mt1/2),int(Mp2/2)]))
np.linalg.norm(fgrad)

p1v, t1v =  np.meshgrid(phi1, theta1)
x1 = np.zeros([Mt1*Mp1,2])
t1v.shape
for i in range(Mt1*Mp1):
    x1[i,0] = t1v.ravel()[i]
    x1[i,1] = p1v.ravel()[i]

nfsft_plan = nfft.nfsft.plan(Mt1*Mp1,N)
nfsft_plan.set_local_coords(x1)
f_hat1 = nfft.nfsft.SphericalFourierCoefficients(N)
f_hat1[2,0] = 1
f1 = nfsft_plan.compute_Ymatrix_multiplication(f_hat1)
pl.imshow(np.real(f1.reshape(Mt1,Mp1)))
pl.imshow(np.real(f.reshape(Mt1,Mp1,Mt2,Mp2)[:,:,0,0]))

f1.reshape(Mt1,Mp1)[10,8]
f.reshape(Mt1,Mp1,Mt2,Mp2)[10,8,0,0]
np.sqrt(4*np.pi)
np.linalg.norm(f1.reshape(Mt1,Mp1))
np.sqrt(4*np.pi)*np.linalg.norm(f.reshape(Mt1,Mp1,Mt2,Mp2)[:,:,0,0])


test_fhat = np.zeros((N+1)**4,dtype=np.complex)
i = 0
for k1 in range(-1,1+1):
    for k2 in range(-1,1+1):
        for m1 in range(abs(k1):1):
            for m2 in range(abs())
        i+=1
        print(i)

nfft.nfdsft.linearized_index(1,-1,2,0,1)
abs(-1)


N=2
fhat = nfft.nfdsft.DoubleSphericalFourierCoefficients(N)
i = 0
for M0 in range(0,N+1):
    for M1 in range(0,N+1):
        for K0 in range(-M0,M0+1):
            for K1 in range(-M1,M1+1):
                #print(nfft.nfdsft.linearized_index(M0,M1,K0,K1))
                fhat[M0,M1,K0,K1] = 1#nfft.nfdsft.linearized_index(M0,M1,K0,K1)
#M0,M1 = 2,1
#fhat[M0,M1,-M0:M0,-M1:M1] = 0
temp = np.zeros([2*M0+1,2*M1+1],dtype=np.complex)
#for K0 in range(-M0,M0+1):
#    for K1 in range(-M1,M1+1):
        temp[K0+M0,K1+M1]=fhat[M0,M1,K0,K1]
temp
fhat[M0,M1,-M0:M0,-M1:M1]
nfdsft_plan = nfft.nfdsft.plan(1,N)
#fhat.array[:]=0
#fhat[0,0,0,0]=1
fhat_mult = nfdsft_plan._compute_Dtheta1_matrix_adjoint_multiplication(fhat)
fhat[0,0,:,:]
fhat_mult[1,1,:,:]
fhat_mult[1,0,:,:]
fhat_mult[0,1,:,:]
nfft.cnfdsft._set_f_hat_internal(nfdsft_plan,fhat)

nfdsft_plan._f_hat_internal
i=0
N+=1
for K0 in range(-N,N+1):
    for K1 in range(-N,N+1):
        for M0 in range(abs(K0),N+1):
            for M1 in range(abs(K1),N+1):
                #print(nfft.nfdsft.linearized_index(M0,M1,K0,K1))
                nfdsft_plan._f_hat_internal[i] = nfft.nfdsft.linearized_index(M0,M1,K0,K1)
                i+=1
np.sort(nfdsft_plan._f_hat_internal)

nfft.cnfdsft._get_from_f_hat_internal(nfdsft_plan,fhat)
fhat.array

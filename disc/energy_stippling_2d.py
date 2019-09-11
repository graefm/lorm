import lorm
from lorm.manif import EuclideanSpace
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfft
import numpy as np
import copy as cp

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N):
        '''
        plan for computing the (polynomial) L^2 discrepancy for points measures on the E2 (2d-Torus)
        M - number of points
        N - polynomial degree
        '''
        self._M = M
        self._N = N
        self._nfft_plan = nfft.NFFT2D(M,N,N)
        self._lambda_hat = np.zeros([N,N])
        for i in range(1,N):
            for j in range(1,N):
                norm_squared = (i-N/2)**2+(j-N/2)**2
                self._lambda_hat[i,j] = 1./(np.power(1+norm_squared,3/2))
        self._mu_hat = np.zeros([N,N],dtype=np.complex)
        self._mu_hat[int(N/2),int(N/2)] = 1
        self._weights = np.ones(M,dtype=np.float)/M

        def f(point_array_coords):
            err_vec = self._eval_error_vector(point_array_coords)
            return np.real(np.sum(err_vec*err_vec.conjugate()*self._lambda_hat))

        def grad(point_array_coords):
            return np.real(self._eval_grad_error_vector(point_array_coords))

    #    def hess_mult(base_point_array_coords, tangent_array_coords):
    #        # approximation
    #        norm = np.linalg.norm(tangent_array_coords)
    #        h = 1e-10
    #        return norm*(grad(base_point_array_coords + h*tangent_array_coords/norm) - grad(base_point_array_coords))/h

        ManifoldObjectiveFunction.__init__(self,lorm.manif.EuclideanSpace(2),f,grad=grad,hess_mult=None, parameterized=False)

    def hess_mult(self,tangent_vector_array):
        hess_mult_vector_array = cp.deepcopy(tangent_vector_array)
        base_point_array_coords = hess_mult_vector_array.base_point_array.coords
        tangent_array_coords = tangent_vector_array.coords
        norm = np.linalg.norm(tangent_array_coords)
        h = 1e-7
        hess_mult_vector_array.coords[:] = norm*np.real(self._eval_grad_error_vector(base_point_array_coords + h*tangent_array_coords/norm) - self._eval_grad_error_vector(base_point_array_coords))/h
        return hess_mult_vector_array


    def _eval_error_vector(self,point_array_coords):
        self._nfft_plan.x = np.mod(point_array_coords+0.5,1)-0.5
        self._nfft_plan.precompute_x()
        self._nfft_plan.f[:] = self._weights
        self._nfft_plan.adjoint()
        err_vec = np.zeros([self._N,self._N],dtype=np.complex)
        err_vec[:] = self._nfft_plan.f_hat[:] - self._mu_hat[:]
        return err_vec

    def _eval_grad_error_vector(self,point_array_coords):
        #self._nfft_plan.x = np.mod(point_array_coords+0.5,1)-0.5
        #self._nfft_plan.precompute_x()
        grad =np.zeros([self._M,2],dtype=np.complex)

        err_vec = self._eval_error_vector(point_array_coords) * self._lambda_hat[:]
        #dx
        self._nfft_plan.f_hat[:] = err_vec[:]
        for i in range(self._N):
            self._nfft_plan.f_hat[i,:] *= -2*np.pi*1j*(i-self._N/2)
        self._nfft_plan.trafo()
        grad[:,0] = 2*self._weights*self._nfft_plan.f[:]

        #dy
        self._nfft_plan.f_hat[:] = err_vec[:]
        for i in range(self._N):
            self._nfft_plan.f_hat[:,i] *= -2*np.pi*1j*(i-self._N/2)
        self._nfft_plan.trafo()
        grad[:,1] = 2*self._weights*self._nfft_plan.f[:]
        return grad

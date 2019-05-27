import lorm
from lorm.manif import Sphere2
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfsft
import numpy as np
import copy as cp

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N, alpha, p):
        '''
        plan for computing the (polynomial) L^2 discrepancy of curves on the sphere S^2
        M - number of points
        N - polynomial degree
        '''
        self._M = M
        self._N = N
        self._nfsft_plan = nfsft.plan(M, N+2)
        self._lambda_hat = nfsft.SphericalFourierCoefficients(N)
        self._lambda_hat.array[:] = 1
        self._mu_hat = nfsft.SphericalFourierCoefficients(N)
        self._mu_hat[0,0] = 1
        self._weights = np.sqrt(4*np.pi) * np.ones([M,1],dtype=float) / M
        self._alpha = alpha
        self._p = p


        def f(point_array_coords):
            lengths = self._eval_lengths(point_array_coords)
            err_vector = self._eval_error_vector(point_array_coords)
            return np.sum(np.real(np.dot(err_vector.array.conjugate(),self._lambda_hat.array*err_vector.array))) + self._alpha*np.power((self._M)**(self._p-1)*np.sum(lengths**(self._p)),1/self._p)

        def grad(point_array_coords):
            le  = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2*np.real(self._nfsft_plan.compute_gradYmatrix_multiplication(le)) * self._weights + self._alpha*self._eval_grad_sum_lengths_squared_powers(point_array_coords)
            return grad

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-7
            return norm*(self._grad(base_point_array_coords + h*tangent_array_coords/norm) - self._grad(base_point_array_coords))/h


        ManifoldObjectiveFunction.__init__(self,Sphere2(),f,grad=grad,hess_mult=hess_mult, parameterized=True)

    def _eval_error_vector(self,point_array_coords):
        self._nfsft_plan.set_local_coords(point_array_coords)
        err_vector = self._nfsft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

    def _eval_lengths(self,point_array_coords):
        lengths = np.zeros([self._M])
        theta = point_array_coords[:,0]
        dp = np.zeros([self._M])
        dp[:] = point_array_coords[:,1]
        dp[0] -= point_array_coords[self._M-1,1]
        dp[1:self._M] -= point_array_coords[0:self._M-1,1]
        ct = np.cos(theta)
        st = np.sin(theta)
        #lengths[0] =  np.sqrt(2 * (1 - np.cos(theta[0])*np.cos(theta[self._M-1]) - np.cos(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.sin(theta[self._M-1])))
        lengths[0] =  np.sqrt(2 * (1 - ct[0]*ct[self._M-1] - np.cos(dp[0])*st[0]*st[self._M-1]))
        #for i in range(1,self._M):
        #    lengths[i] =  np.sqrt(2 * (1 - np.cos(theta[i])*np.cos(theta[i-1]) - np.cos(phi[i]-phi[i-1])*np.sin(theta[i])*np.sin(theta[i-1])))
        lengths[1:self._M] =  np.sqrt(2 * (1 - ct[1:self._M]*ct[0:self._M-1]- np.cos(dp[1:self._M])*st[1:self._M]*st[0:self._M-1]))

        return lengths

    def _eval_grad_lengths1(self,point_array_coords):
        grad_lengths = np.zeros([self._M,2])
        theta = point_array_coords[:,0]
        dp = np.zeros([self._M])
        dp[:] = point_array_coords[:,1]
        dp[0] -= point_array_coords[self._M-1,1]
        dp[1:self._M] -= point_array_coords[0:self._M-1,1]
        ct = np.cos(theta)
        st = np.sin(theta)
        lengths = self._eval_lengths(point_array_coords)
        #lengths = np.sqrt(2 * (1 - np.cos(theta[0])*np.cos(theta[self._M-1]) - np.cos(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.sin(theta[self._M-1])))
        #grad_lengths[0,0] =  (np.sin(theta[0])*np.cos(theta[self._M-1]) - np.cos(phi[0]-phi[self._M-1])*np.cos(theta[0])*np.sin(theta[self._M-1]))/lengths
        grad_lengths[0,0] =  (st[0]*ct[self._M-1] - np.cos(dp[0])*ct[0]*st[self._M-1])/lengths[0]
        #grad_lengths[0,1] =  np.sin(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.sin(theta[self._M-1])/lengths
        grad_lengths[0,1] =  np.sin(dp[0])*st[0]*st[self._M-1]/lengths[0]
        #for i in range(1,self._M):
        #    lengths = np.sqrt(2 * (1 - np.cos(theta[i])*np.cos(theta[i-1]) - np.cos(phi[i]-phi[i-1])*np.sin(theta[i])*np.sin(theta[i-1])))
        #    grad_lengths[i,0] =  (np.sin(theta[i])*np.cos(theta[i-1]) - np.cos(phi[i]-phi[i-1])*np.cos(theta[i])*np.sin(theta[i-1]))/lengths
        #    grad_lengths[i,1] =  np.sin(phi[i]-phi[i-1])*np.sin(theta[i])*np.sin(theta[i-1])/lengths
        grad_lengths[1:self._M,0] = (st[1:self._M]*ct[0:self._M-1] - np.cos(dp[1:self._M])*ct[1:self._M]*st[0:self._M-1])/lengths[1:self._M]
        grad_lengths[1:self._M,1] = np.sin(dp[1:self._M])*st[1:self._M]*st[0:self._M-1]/lengths[1:self._M]
        return grad_lengths

    def _eval_grad_lengths2(self,point_array_coords):
        grad_lengths = np.zeros([self._M,2])
        theta = point_array_coords[:,0]
        dp = np.zeros([self._M])
        dp[:] = point_array_coords[:,1]
        dp[0] -= point_array_coords[self._M-1,1]
        dp[1:self._M] -= point_array_coords[0:self._M-1,1]
        ct = np.cos(theta)
        st = np.sin(theta)
        lengths = self._eval_lengths(point_array_coords)
        #phi = point_array_coords[:,1]
        #lengths = np.sqrt(2 * (1 - np.cos(theta[0])*np.cos(theta[self._M-1]) - np.cos(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.sin(theta[self._M-1])))
        #grad_lengths[self._M-1,0] =  (np.cos(theta[0])*np.sin(theta[self._M-1]) - np.cos(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.cos(theta[self._M-1]))/lengths
        grad_lengths[self._M-1,0] =  (ct[0]*st[self._M-1] - np.cos(dp[0])*st[0]*ct[self._M-1])/lengths[0]
        #grad_lengths[self._M-1,1] =  -np.sin(phi[0]-phi[self._M-1])*np.sin(theta[0])*np.sin(theta[self._M-1])/lengths
        grad_lengths[self._M-1,1] =  -np.sin(dp[0])*st[0]*st[self._M-1]/lengths[0]
        #for i in range(0,self._M-1):
        #    lengths = np.sqrt(2 * (1 - np.cos(theta[i+1])*np.cos(theta[i]) - np.cos(phi[i+1]-phi[i])*np.sin(theta[i+1])*np.sin(theta[i])))
        #    grad_lengths[i,0] =  (np.cos(theta[i+1])*np.sin(theta[i]) - np.cos(phi[i+1]-phi[i])*np.sin(theta[i+1])*np.cos(theta[i]))/lengths
        #    grad_lengths[i,1] =  -np.sin(phi[i+1]-phi[i])*np.sin(theta[i+1])*np.sin(theta[i])/lengths
        grad_lengths[0:self._M-1,0] =  (ct[1:self._M]*st[0:self._M-1] - np.cos(dp[1:self._M])*st[1:self._M]*ct[0:self._M-1])/lengths[1:self._M]
        grad_lengths[0:self._M-1,1] =  -np.sin(dp[1:self._M])*st[1:self._M]*st[0:self._M-1]/lengths[1:self._M]
        return grad_lengths

    def _eval_grad_sum_lengths_squared_powers(self,point_array_coords):
        lengths = self._eval_lengths(point_array_coords).reshape([self._M,1])
        sum_lengths_power = 1/self._p*np.power((self._M)**(self._p-1)*np.sum(lengths**(self._p)),1/self._p-1)
        grad_lengths1 = self._eval_grad_lengths1(point_array_coords)
        grad_lengths2 = self._eval_grad_lengths2(point_array_coords)
        grad = np.zeros((self._M,2))
        grad[0,:] +=  self._p*lengths[0]**(self._p-1)*grad_lengths1[0,:]
        #for i in range(1,self._M):
        #    grad[i,:] += self._p*lengths[i]**(self._p-1)*grad_lengths1[i,:]
        grad[1:self._M,:] += self._p*lengths[1:self._M]**(self._p-1)*grad_lengths1[1:self._M,:]
        grad[self._M-1,:] += self._p*lengths[0]**(self._p-1)*grad_lengths2[self._M-1,:]
        #for i in range(0,self._M-1):
        #    grad[i,:] += self._p*lengths[i+1]**(self._p-1)*grad_lengths2[i,:]
        grad[0:self._M-1,:] += self._p*lengths[1:self._M]**(self._p-1)*grad_lengths2[0:self._M-1,:]
        return sum_lengths_power*(self._M)**(self._p-1)*grad

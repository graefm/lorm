import lorm
from lorm.manif import SO3
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfsoft
import numpy as np

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N, alpha, L, equality_constraint=False):
        '''
        plan for computing the (polynomial) L^2 discrepancy on the rotation group SO(3)
        M - number of points
        N - polynomial degree
        '''
        self._M = M
        self._N = N
        self._m = 5
        self._alpha = alpha
        self._L = L
        self._equality_constraint = equality_constraint
        self._nfsoft_plan = nfsoft.plan(M, N+2, self._m)
        self._lambda_hat = nfsoft.SO3FourierCoefficients(N)
        for n in range(N+1):
            self._lambda_hat[n,:,:] = (2.*n+1)/((2.*n-1)*(2.*n+1)**2*(2.*n+3))
        self._mu_hat = nfsoft.SO3FourierCoefficients(N)
        self._mu_hat[0,0,0] = 8*np.pi**2 # int_SO(3) D^n_k,k'(x) mu(x)
        self._weights = 8*np.pi**2 * np.ones([M,1],dtype=float) / M

        def f(point_array_coords):
            lengths = self._eval_lengths(point_array_coords)
            err_vec = self._eval_error_vector(point_array_coords)
            pos_diff_lengths = self._M*lengths-self._L
            if self._equality_constraint == False:
                pos_diff_lengths[ pos_diff_lengths < 0] = 0
            return  np.sum(np.real(np.dot(err_vec.array.conjugate(),self._lambda_hat.array*err_vec.array))) + self._alpha*1./self._M*np.sum(pos_diff_lengths**2)

        def grad(point_array_coords):
            le  = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2*np.real(self._nfsoft_plan.compute_gradYmatrix_multiplication(le)) * self._weights + self._alpha*1./self._M*self._eval_grad_sum_lengths_squared(point_array_coords)
            return grad

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-12
            return norm*(self._grad(base_point_array_coords + h*tangent_array_coords/norm) - self._grad(base_point_array_coords))/h

        ManifoldObjectiveFunction.__init__(self,SO3(),f,grad=grad,hess_mult=hess_mult, parameterized=True)

    def _eval_error_vector(self,point_array_coords):
        self._nfsoft_plan.set_local_coords(point_array_coords)
        err_vector = self._nfsoft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

    def _eval_cos_signed_lengths_half(self,point_array_coords):
        lengths = np.zeros([self._M])
        dp = np.zeros([self._M,3])
        dp[:,:] = point_array_coords[:,:]
        dp[0,:] -= point_array_coords[self._M-1,:]
        dp[1:self._M,:] -= point_array_coords[0:self._M-1,:]
        st = np.zeros([self._M])
        st[:] = point_array_coords[:,1]
        st[0] += point_array_coords[self._M-1,1]
        st[1:self._M] += point_array_coords[0:self._M-1,1]
        return np.cos(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.cos(dp[:,1]/2)-np.sin(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.cos(st/2)

    def _eval_lengths(self,point_array_coords):
        return 2*np.arccos(np.abs(self._eval_cos_signed_lengths_half(point_array_coords)))

    def _eval_grad_lengths1(self,point_array_coords):
        dp = np.zeros([self._M,3])
        dp[:,:] = point_array_coords[:,:]
        dp[0,:] -= point_array_coords[self._M-1,:]
        dp[1:self._M,:] -= point_array_coords[0:self._M-1,:]
        st = np.zeros([self._M])
        st[:] = point_array_coords[:,1]
        st[0] += point_array_coords[self._M-1,1]
        st[1:self._M] += point_array_coords[0:self._M-1,1]

        cos_signed_lengths_half = self._eval_cos_signed_lengths_half(point_array_coords)
        temp = np.zeros([self._M,1])
        temp[:,0] = -np.sign(cos_signed_lengths_half)/np.sqrt(1-np.power(cos_signed_lengths_half,2))

        grad_lengths = np.zeros([self._M,3])
        grad_lengths[:,0] = (-np.sin(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.cos(dp[:,1]/2)-np.cos(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.cos(st/2))
        grad_lengths[:,1] = (-np.cos(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.sin(dp[:,1]/2)+np.sin(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.sin(st/2))
        grad_lengths[:,2] = (-np.cos(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.cos(dp[:,1]/2)-np.sin(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.cos(st/2))
        return grad_lengths*temp

    def _eval_grad_lengths2(self,point_array_coords):
        dp = np.empty([self._M,3])
        dp[:,:] = point_array_coords[:,:]
        dp[0,:] -= point_array_coords[self._M-1,:]
        dp[1:self._M,:] -= point_array_coords[0:self._M-1,:]
        st = np.zeros([self._M])
        st[:] = point_array_coords[:,1]
        st[0] += point_array_coords[self._M-1,1]
        st[1:self._M] += point_array_coords[0:self._M-1,1]

        cos_signed_lengths_half = self._eval_cos_signed_lengths_half(point_array_coords)
        temp = np.empty([self._M,1])
        temp[:,0] = -np.sign(cos_signed_lengths_half)/np.sqrt(1-np.power(cos_signed_lengths_half,2))

        grad_lengths = np.empty([self._M,3])
        grad_lengths[:,0] = (np.sin(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.cos(dp[:,1]/2)+np.cos(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.cos(st/2))
        grad_lengths[:,1] = (np.cos(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.sin(dp[:,1]/2)+np.sin(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.sin(st/2))
        grad_lengths[:,2] = (np.cos(dp[:,0]/2)*np.sin(dp[:,2]/2)*np.cos(dp[:,1]/2)+np.sin(dp[:,0]/2)*np.cos(dp[:,2]/2)*np.cos(st/2))
        return grad_lengths*temp

    def _eval_grad_sum_lengths_squared(self,point_array_coords):
        lengths = self._eval_lengths(point_array_coords).reshape([self._M,1])
        grad_lengths1 = self._eval_grad_lengths1(point_array_coords)
        grad_lengths2 = self._eval_grad_lengths2(point_array_coords)

        grad = np.zeros((self._M,3))
        pos_diff_lengths = self._M*lengths-self._L
        if self._equality_constraint == False:
            pos_diff_lengths[ pos_diff_lengths < 0] = 0
        grad += pos_diff_lengths*grad_lengths1
        grad[0:self._M-1,:] += pos_diff_lengths[1:self._M]*grad_lengths2[1:self._M,:]
        grad[self._M-1,:] += pos_diff_lengths[0]*grad_lengths2[0,:]
        return 2*self._M*grad

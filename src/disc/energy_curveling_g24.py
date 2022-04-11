import lorm
from lorm.manif import Sphere2
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfdsft
import numpy as np

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N, alpha, L, m=5, sigma=2, equality_constraint = False):
        '''
        plan for computing the (polynomial) L^2 discrepancy on the Grassmannian G_{2,4} = S^2 x S^2 / {-1}
        M - number of points on S^2 (must be even!)
        N - polynomial degree
        '''
        self._M = M
        self._N = N
        self._m = m
        self._sigma = sigma
        self._nfdsft_plan = nfdsft.plan(M, N+2, m=self._m, sigma=self._sigma)
        self._lambda_hat = nfdsft.DoubleSphericalFourierCoefficients(N)
        #self._lambda_hat.array[:] = 1 # for s^2 x S^2
        for M0 in range(N+1):
            for M1 in range(N+1):
                #l1 = (M0+M1)/2.
                #l2 = np.abs(M0-M1)/2.
                l_squared = (M0**2+M1**2)/2. #l1**2+l2**2
                if M0 % 2 == M1 % 2:
                    self._lambda_hat[M0,M1,:,:] = 1./(l_squared+1)**(5./2.)
        self._mu_hat = nfdsft.DoubleSphericalFourierCoefficients(N)
        self._mu_hat[0,0,0,0] = 1
        self._weights = 4*np.pi*np.ones([M,1],dtype=float) / self._M
        self._alpha = alpha
        self._L = L
        self._equality_constraint = equality_constraint

        def f(point_array_coords):
            err_vector = self._eval_error_vector(point_array_coords)
            lengths = self._eval_lengths(point_array_coords)
            pos_diff_lengths = self._M*lengths-self._L
            if self._equality_constraint == False:
                pos_diff_lengths[ pos_diff_lengths < 0] = 0
            err_vector = self._eval_error_vector(point_array_coords)
            return np.sum(np.real(np.dot(err_vector.array.conjugate(),self._lambda_hat.array*err_vector.array))) + self._alpha/self._M*np.sum(pos_diff_lengths**2)

        def grad(point_array_coords):
            le  = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2*np.real(self._nfdsft_plan.compute_gradYmatrix_multiplication(le)) * self._weights
            return grad.reshape(2*self._M,2) + self._alpha/self._M*self._eval_grad_sum_lengths_squared(point_array_coords).reshape(2*self._M,2)

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-12
            return norm*(self._grad(base_point_array_coords + h*tangent_array_coords/norm) - self._grad(base_point_array_coords))/h

        ManifoldObjectiveFunction.__init__(self,Sphere2(),f,grad=grad,hess_mult=hess_mult, parameterized=True)


    def _eval_error_vector(self,point_array_coords):
        self._nfdsft_plan.set_local_coords(point_array_coords.reshape(self._M,4))
        err_vector = self._nfdsft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

    def _eval_lengths(self,point_array_coords):
        coords = point_array_coords.reshape(self._M,4)
        theta1 = coords[:,0]
        theta2 = coords[:,2]
        dp1 = np.zeros([self._M])
        dp2 = np.zeros([self._M])
        dp1[:] = coords[:,1]
        dp2[:] = coords[:,3]
        dp1[0] -= coords[self._M-1,1]
        dp2[0] -= coords[self._M-1,3]
        dp1[1:self._M] -= coords[0:self._M-1,1]
        dp2[1:self._M] -= coords[0:self._M-1,3]
        ct1 = np.cos(theta1)
        ct2 = np.cos(theta2)
        st1 = np.sin(theta1)
        st2 = np.sin(theta2)
        lengths1_squared = np.zeros([self._M])
        lengths2_squared = np.zeros([self._M])
        lengths1_squared[0] = 2 * (1 - ct1[0]*ct1[self._M-1] - np.cos(dp1[0])*st1[0]*st1[self._M-1])
        lengths2_squared[0] = 2 * (1 - ct2[0]*ct2[self._M-1] - np.cos(dp2[0])*st2[0]*st2[self._M-1])
        lengths1_squared[1:self._M] =  2 * (1 - ct1[1:self._M]*ct1[0:self._M-1]- np.cos(dp1[1:self._M])*st1[1:self._M]*st1[0:self._M-1])
        lengths2_squared[1:self._M] =  2 * (1 - ct2[1:self._M]*ct2[0:self._M-1]- np.cos(dp2[1:self._M])*st2[1:self._M]*st2[0:self._M-1])
        return np.sqrt(lengths1_squared + lengths2_squared)


    def _eval_grad_lengths1(self,point_array_coords):
        coords = point_array_coords.reshape(self._M,4)
        theta1 = coords[:,0]
        theta2 = coords[:,2]
        dp1 = np.zeros([self._M])
        dp2 = np.zeros([self._M])
        dp1[:] = coords[:,1]
        dp2[:] = coords[:,3]
        dp1[0] -= coords[self._M-1,1]
        dp2[0] -= coords[self._M-1,3]
        dp1[1:self._M] -= coords[0:self._M-1,1]
        dp2[1:self._M] -= coords[0:self._M-1,3]
        ct1 = np.cos(theta1)
        ct2 = np.cos(theta2)
        st1 = np.sin(theta1)
        st2 = np.sin(theta2)
        lengths = self._eval_lengths(point_array_coords)
        grad_lengths = np.zeros([self._M,4])
        grad_lengths[0,0] =  (st1[0]*ct1[self._M-1] - np.cos(dp1[0])*ct1[0]*st1[self._M-1])/lengths[0]
        grad_lengths[0,2] =  (st2[0]*ct2[self._M-1] - np.cos(dp2[0])*ct2[0]*st2[self._M-1])/lengths[0]
        grad_lengths[0,1] =  np.sin(dp1[0])*st1[0]*st1[self._M-1]/lengths[0]
        grad_lengths[0,3] =  np.sin(dp2[0])*st2[0]*st2[self._M-1]/lengths[0]
        grad_lengths[1:self._M,0] = (st1[1:self._M]*ct1[0:self._M-1] - np.cos(dp1[1:self._M])*ct1[1:self._M]*st1[0:self._M-1])/lengths[1:self._M]
        grad_lengths[1:self._M,2] = (st2[1:self._M]*ct2[0:self._M-1] - np.cos(dp2[1:self._M])*ct2[1:self._M]*st2[0:self._M-1])/lengths[1:self._M]
        grad_lengths[1:self._M,1] = np.sin(dp1[1:self._M])*st1[1:self._M]*st1[0:self._M-1]/lengths[1:self._M]
        grad_lengths[1:self._M,3] = np.sin(dp2[1:self._M])*st2[1:self._M]*st2[0:self._M-1]/lengths[1:self._M]
        return grad_lengths

    def _eval_grad_lengths2(self,point_array_coords):
        coords = point_array_coords.reshape(self._M,4)
        theta1 = coords[:,0]
        theta2 = coords[:,2]
        dp1 = np.zeros([self._M])
        dp2 = np.zeros([self._M])
        dp1[:] = coords[:,1]
        dp2[:] = coords[:,3]
        dp1[0] -= coords[self._M-1,1]
        dp2[0] -= coords[self._M-1,3]
        dp1[1:self._M] -= coords[0:self._M-1,1]
        dp2[1:self._M] -= coords[0:self._M-1,3]
        ct1 = np.cos(theta1)
        ct2 = np.cos(theta2)
        st1 = np.sin(theta1)
        st2 = np.sin(theta2)
        lengths = self._eval_lengths(point_array_coords)
        grad_lengths = np.zeros([self._M,4])
        grad_lengths[self._M-1,0] =  (ct1[0]*st1[self._M-1] - np.cos(dp1[0])*st1[0]*ct1[self._M-1])/lengths[0]
        grad_lengths[self._M-1,2] =  (ct2[0]*st2[self._M-1] - np.cos(dp2[0])*st2[0]*ct2[self._M-1])/lengths[0]
        grad_lengths[self._M-1,1] =  -np.sin(dp1[0])*st1[0]*st1[self._M-1]/lengths[0]
        grad_lengths[self._M-1,3] =  -np.sin(dp2[0])*st2[0]*st2[self._M-1]/lengths[0]
        grad_lengths[0:self._M-1,0] =  (ct1[1:self._M]*st1[0:self._M-1] - np.cos(dp1[1:self._M])*st1[1:self._M]*ct1[0:self._M-1])/lengths[1:self._M]
        grad_lengths[0:self._M-1,2] =  (ct2[1:self._M]*st2[0:self._M-1] - np.cos(dp2[1:self._M])*st2[1:self._M]*ct2[0:self._M-1])/lengths[1:self._M]
        grad_lengths[0:self._M-1,1] =  -np.sin(dp1[1:self._M])*st1[1:self._M]*st1[0:self._M-1]/lengths[1:self._M]
        grad_lengths[0:self._M-1,3] =  -np.sin(dp2[1:self._M])*st2[1:self._M]*st2[0:self._M-1]/lengths[1:self._M]
        return grad_lengths

    def _eval_grad_sum_lengths_squared(self,point_array_coords):
        lengths = self._eval_lengths(point_array_coords).reshape([self._M,1])
        grad_lengths1 = self._eval_grad_lengths1(point_array_coords)
        grad_lengths2 = self._eval_grad_lengths2(point_array_coords)
        grad = np.zeros((self._M,4))
        pos_diff_lengths = self._M*lengths-self._L
        if self._equality_constraint == False:
            pos_diff_lengths[ pos_diff_lengths < 0] = 0
        grad[0,:] +=  pos_diff_lengths[0]*grad_lengths1[0,:]
        grad[1:self._M,:] += pos_diff_lengths[1:self._M]*grad_lengths1[1:self._M,:]
        grad[self._M-1,:] += pos_diff_lengths[0]*grad_lengths2[self._M-1,:]
        grad[0:self._M-1,:] += pos_diff_lengths[1:self._M]*grad_lengths2[0:self._M-1,:]
        return 2*self._M*grad

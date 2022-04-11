import lorm
from lorm.manif import Sphere2
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfdsft
import numpy as np

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N, m=5, sigma=2):
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
                    self._lambda_hat[M0,M1,:,:] = 1.#/(l_squared+1)**(0.5)
        self._mu_hat = nfdsft.DoubleSphericalFourierCoefficients(N)
        self._mu_hat[0,0,0,0] = 1
        self._weights = 4*np.pi*np.ones([M,1],dtype=float) / M

        def f(point_array_coords):
            err_vector = self._eval_error_vector(point_array_coords)
            return np.sum(np.real(np.dot(err_vector.array.conjugate(),self._lambda_hat.array*err_vector.array)))

        def grad(point_array_coords):
            le  = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2*np.real(self._nfdsft_plan.compute_gradYmatrix_multiplication(le)) * self._weights
            return grad.reshape(2*self._M,2)

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-7
            return norm*(self._grad(base_point_array_coords + h*tangent_array_coords/norm) - self._grad(base_point_array_coords))/h

        ManifoldObjectiveFunction.__init__(self,Sphere2(),f,grad=grad,hess_mult=hess_mult, parameterized=True)


    def _eval_error_vector(self,point_array_coords):
        self._nfdsft_plan.set_local_coords(point_array_coords.reshape(self._M,4))
        err_vector = self._nfdsft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

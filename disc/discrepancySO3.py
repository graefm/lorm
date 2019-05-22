import lorm
from lorm.manif import SO3
from lorm.funcs import ManifoldObjectiveFunction
import numpy as np

class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N, m=5):
        '''
        plan for computing the (polynomial) L^2 discrepancy on the rotation group SO(3)
        M - number of points
        N - polynomial degree
        '''
        self._M = M
        self._N = N
        self._m = m
        self._nfsoft_plan = lorm.nfsoft.plan(M, N+2, self._m)
        self._lambda_hat = lorm.nfsoft.SO3FourierCoefficients(N)
        for n in range(N+1):
            self._lambda_hat[n,:,:] = (2.*n+1)/(8*np.pi**2)
        self._mu_hat = lorm.nfsoft.SO3FourierCoefficients(N)
        self._mu_hat[0,0,0] = 8*np.pi**2 # int_SO(3) D^n_k,k'(x) mu(x)
        self._weights = 8*np.pi**2 * np.ones([M,1],dtype=float) / M

        def f(point_array_coords):
            err_vector = self._eval_error_vector(point_array_coords)
            return np.sum(np.real(np.dot(err_vector.array.conjugate(),self._lambda_hat.array*err_vector.array)))

        def grad(point_array_coords):
            le  = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2*np.real(self._nfsoft_plan.compute_gradYmatrix_multiplication(le)) * self._weights
            return grad

        def hess_mult(base_point_array_coords, tangent_array_coords):
            le  = self._eval_error_vector(base_point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            hess_mult = 2*np.real(self._nfsoft_plan.compute_gradYmatrix_multiplication(\
                                  self._lambda_hat * self._nfsoft_plan.compute_gradYmatrix_adjoint_multiplication(\
                                  tangent_array_coords * self._weights )) ) * self._weights
            hess_array = np.real(self._nfsoft_plan.compute_hessYmatrix_multiplication(le))
            hess_mult[:,0] += 2 * self._weights.reshape(self._M) * (  hess_array[:,0] * tangent_array_coords[:,0]\
                                                                    + hess_array[:,1] * tangent_array_coords[:,1]\
                                                                    + hess_array[:,2] * tangent_array_coords[:,2])
            hess_mult[:,1] += 2 * self._weights.reshape(self._M) * (   hess_array[:,1] * tangent_array_coords[:,0]\
                                                                     + hess_array[:,3] * tangent_array_coords[:,1]\
                                                                     + hess_array[:,4] * tangent_array_coords[:,2])
            hess_mult[:,2] += 2 * self._weights.reshape(self._M) * (   hess_array[:,2] * tangent_array_coords[:,0]\
                                                                     + hess_array[:,4] * tangent_array_coords[:,1]\
                                                                     + hess_array[:,5] * tangent_array_coords[:,2])
            return hess_mult

        ManifoldObjectiveFunction.__init__(self,SO3(),f,grad=grad,hess_mult=hess_mult, parameterized=True)


    def _eval_error_vector(self,point_array_coords):
        self._nfsoft_plan.set_local_coords(point_array_coords)
        err_vector = self._nfsoft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

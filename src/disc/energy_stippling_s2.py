from lorm.manif import Sphere2
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfsft
import numpy as np


class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N):
        """
        plan for computing the (polynomial) L^2 discrepancy on the sphere S^2
        M - number of points
        N - polynomial degree
        """
        self._M = M
        self._N = N
        self._nfsft_plan = nfsft.plan(M, N)
        self._lambda_hat = nfsft.SphericalFourierCoefficients(N)
        self._lambda_hat.array[:] = 1
        self._mu_hat = nfsft.SphericalFourierCoefficients(N)
        self._mu_hat[0, 0] = 1
        self._weights = np.sqrt(4 * np.pi) * np.ones([M, 1], dtype=float) / M

        def f(point_array_coords):
            err_vector = self._eval_error_vector(point_array_coords)
            return np.sum(np.real(np.dot(err_vector.array.conjugate(), self._lambda_hat.array * err_vector.array)))

        def grad(point_array_coords):
            le = self._eval_error_vector(point_array_coords)
            le *= self._lambda_hat
            # we already set the point_array_coords in _eval_error_vector
            grad = 2 * np.real(self._nfsft_plan.compute_gradYmatrix_multiplication(le)) * self._weights
            return grad

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-7
            return norm * (self._grad(base_point_array_coords + h * tangent_array_coords / norm) - self._grad(
                base_point_array_coords)) / h

        ManifoldObjectiveFunction.__init__(self, Sphere2(), f, grad=grad, hess_mult=hess_mult, parameterized=True)

    def _eval_error_vector(self, point_array_coords):
        self._nfsft_plan.set_local_coords(point_array_coords)
        err_vector = self._nfsft_plan.compute_Ymatrix_adjoint_multiplication(self._weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

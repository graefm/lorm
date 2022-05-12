from lorm.manif import Sphere2, EuclideanSpace, ProductManifold
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
        self._mu_hat = nfsft.SphericalFourierCoefficients(N)
        self._mu_hat[0, 0] = np.sqrt(4 * np.pi)
        # uniform weights = 4 * np.pi / M

        def f(point_array_coords):
            points = point_array_coords[:, 0:2]
            weights = point_array_coords[:, 2].reshape(self._M, 1)
            err_vec = self._eval_error_vector(points, weights)
            return np.sum(np.real(np.dot(err_vec.array.conjugate(), err_vec.array)))

        def grad(point_array_coords):
            points = point_array_coords[:, 0:2]
            weights = point_array_coords[:, 2].reshape(self._M, 1)
            err_vec = self._eval_error_vector(points, weights)
            # we already set the point_array_coords in _eval_error_vector
            grad = np.zeros((point_array_coords.shape[0], 3))  # 2))
            grad[:, 0:2] = 2 * np.real(self._nfsft_plan.compute_gradYmatrix_multiplication(err_vec)) * weights
            grad[:, 2] = 2 * np.real(self._nfsft_plan.compute_Ymatrix_multiplication(err_vec))
            return grad

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-12
            return norm * (self._grad(base_point_array_coords + h * tangent_array_coords / norm)
                           - self._grad(base_point_array_coords)) / h

        ManifoldObjectiveFunction.__init__(self, ProductManifold([Sphere2(), EuclideanSpace(1)]), f, grad=grad,
                                           hess_mult=hess_mult, parameterized=True)

    def _eval_error_vector(self, points, weights):
        self._nfsft_plan.set_local_coords(points)
        err_vector = self._nfsft_plan.compute_Ymatrix_adjoint_multiplication(weights, self._N)
        err_vector -= self._mu_hat
        return err_vector

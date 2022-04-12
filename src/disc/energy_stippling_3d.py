import lorm
from lorm.funcs import ManifoldObjectiveFunction
from nfft import nfft
import numpy as np
import copy as cp


class plan(ManifoldObjectiveFunction):
    def __init__(self, M, N):
        """
        plan for computing the (polynomial) L^2 discrepancy for points measures on the E3 (3d-Torus)
        M - number of points
        N - polynomial degree
        """
        self._M = M
        self._N = N
        self._nfft_plan = nfft.NFFT3D(M, N, N, N)
        self._lambda_hat = np.ones([N, N, N])
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    norm_squared = (i - N / 2) ** 2 + (j - N / 2) ** 2 + (k - N / 2) ** 2
                    self._lambda_hat[i, j, k] = 1. / np.power(norm_squared + 1, 2)
        self._mu_hat = np.zeros([N, N, N], dtype=np.complex)
        self._mu_hat[int(N / 2), int(N / 2), int(N / 2)] = 1
        self._weights = np.ones(M, dtype=np.float) / M

        def f(point_array_coords):
            err_vec = self._eval_error_vector(point_array_coords)
            return np.real(np.sum(err_vec * err_vec.conjugate() * self._lambda_hat))

        def grad(point_array_coords):
            return np.real(self._eval_grad_error_vector(point_array_coords))

        def hess_mult(base_point_array_coords, tangent_array_coords):
            norm = np.linalg.norm(tangent_array_coords)
            h = 1e-10
            return norm * (grad(base_point_array_coords + h * tangent_array_coords / norm)
                           - grad(base_point_array_coords)) / h

        ManifoldObjectiveFunction.__init__(self, lorm.manif.EuclideanSpace(3), f, grad=grad, hess_mult=hess_mult,
                                           parameterized=False)

    def _eval_error_vector(self, point_array_coords):
        self._nfft_plan.x = np.mod(point_array_coords + 0.5, 1) - 0.5
        self._nfft_plan.precompute_x()
        self._nfft_plan.f[:] = self._weights
        self._nfft_plan.adjoint()
        err_vec = np.zeros([self._N, self._N, self._N], dtype=np.complex)
        err_vec[:] = self._nfft_plan.f_hat[:] - self._mu_hat[:]
        return err_vec

    def _eval_grad_error_vector(self, point_array_coords):
        grad = np.zeros([self._M, 3], dtype=np.complex)

        err_vec = self._eval_error_vector(point_array_coords) * self._lambda_hat[:]
        # dx
        self._nfft_plan.f_hat[:] = err_vec[:]
        for i in range(self._N):
            self._nfft_plan.f_hat[i, :, :] *= -2 * np.pi * 1j * (i - self._N / 2)
        self._nfft_plan.trafo()
        grad[:, 0] = 2 * self._weights * self._nfft_plan.f[:]

        # dy
        self._nfft_plan.f_hat[:] = err_vec[:]
        for i in range(self._N):
            self._nfft_plan.f_hat[:, i, :] *= -2 * np.pi * 1j * (i - self._N / 2)
        self._nfft_plan.trafo()
        grad[:, 1] = 2 * self._weights * self._nfft_plan.f[:]

        # dz
        self._nfft_plan.f_hat[:] = err_vec[:]
        for i in range(self._N):
            self._nfft_plan.f_hat[:, :, i] *= -2 * np.pi * 1j * (i - self._N / 2)
        self._nfft_plan.trafo()
        grad[:, 2] = 2 * self._weights * self._nfft_plan.f[:]
        return grad

from . import manif
import copy as cp
import numpy as np

from . import cfuncs


def check_manifold(func):
    def assert_equal_manifold(self, obj, *args):
        assert self._manifold == obj.manifold
        return func(self, obj, *args)

    return assert_equal_manifold


class ManifoldObjectiveFunction:
    def __init__(self, manifold, f, grad=None, hess_mult=None, parameterized=False):
        self._manifold = manifold
        self._f = f
        if parameterized:
            assert self._manifold.parameterized
        self._parameterized = parameterized
        if grad is not None:
            self._grad = grad
        if hess_mult is not None:
            self._hess_mult = hess_mult

    @property
    def manifold(self):
        return self._manifold

    @check_manifold
    def f(self, manifold_point_array):
        if not self._parameterized:
            return self._f(manifold_point_array.coords)
        else:
            return self._f(manifold_point_array.local_coords)

    @check_manifold
    def grad(self, manifold_point_array):
        if not self._parameterized:
            tangent_vector_array = manif.TangentVectorArray(manifold_point_array)
            tangent_vector_array.coords = self._grad(manifold_point_array.coords)
            return tangent_vector_array
        else:
            tangent_vector_array = manif.TangentVectorArrayParameterized(manifold_point_array)
            gradfp = self._grad(manifold_point_array.local_coords)
            inverse_riemannian_matrix = self.manifold.compute_inverse_riemannian_matrix(
                manifold_point_array.local_coords)
            grad = np.zeros([gradfp.shape[0], self.manifold.local_dim])
            for i in range(self.manifold.local_dim):
                for j in range(self.manifold.local_dim):
                    grad[:, i] += inverse_riemannian_matrix[:, i, j] * gradfp[:, j]
            tangent_vector_array.local_coords = grad
            return tangent_vector_array

    @check_manifold
    def hess_mult(self, tangent_vector_array):
        hess_mult_vector_array = cp.deepcopy(tangent_vector_array)
        if not self._parameterized:
            base_coords = hess_mult_vector_array.base_point_array.coords
            hess_mult_vector_array.coords = self._hess_mult(base_coords, tangent_vector_array.coords) \
                                            - tangent_vector_array.christoffel_matrix_lin_comb_mult(
                self._grad(base_coords))
            return hess_mult_vector_array
        else:
            base_local_coords = hess_mult_vector_array.base_point_array.local_coords
            tangent_local_coords = tangent_vector_array.local_coords
            hess_mult_local_coords = self._hess_mult(base_local_coords, tangent_local_coords)
            gradfp_local_coords = self._grad(base_local_coords)
            christoffel_matrix_lincomp_param = self.manifold.compute_christoffel_matrix_lin_comb_parameterization(
                base_local_coords, gradfp_local_coords)
            inverse_riemannian_matrix = self.manifold.compute_inverse_riemannian_matrix(base_local_coords)
            for i in range(self.manifold.local_dim):
                for j in range(self.manifold.local_dim):
                    hess_mult_local_coords[:, i] -= christoffel_matrix_lincomp_param[:, i, j] * tangent_local_coords[:,
                                                                                                j]
            hess_mult = np.zeros([hess_mult_local_coords.shape[0], self.manifold.local_dim])
            for i in range(self.manifold.local_dim):
                for j in range(self.manifold.local_dim):
                    hess_mult[:, i] += inverse_riemannian_matrix[:, i, j] * hess_mult_local_coords[:, j]
            hess_mult_vector_array.local_coords = hess_mult
            return hess_mult_vector_array


class PotentialEnergyDotProductKernel(ManifoldObjectiveFunction):
    def __init__(self, manifold, kernel_func, parameterized=False):
        def f(point_array_coords):
            return cfuncs.potential_energy_dot_product_kernel_f(kernel_func, point_array_coords)

        def grad(point_array_coords):
            return cfuncs.potential_energy_dot_product_kernel_grad(kernel_func, point_array_coords)

        def hess_mult(point_array_coords, tangent_array_coords):
            return cfuncs.potential_energy_dot_product_kernel_hess_mult(kernel_func, point_array_coords,
                                                                        tangent_array_coords)

        ManifoldObjectiveFunction.__init__(self, manifold, f, grad=grad, hess_mult=hess_mult,
                                           parameterized=parameterized)


class PotentialEnergySquaredDistanceKernel(ManifoldObjectiveFunction):
    def __init__(self, manifold, kernel_func, parameterized=False):
        def f(point_array_coords):
            return cfuncs.potential_energy_squared_distance_kernel_f(kernel_func, point_array_coords)

        def grad(point_array_coords):
            return cfuncs.potential_energy_squared_distance_kernel_grad(kernel_func, point_array_coords)

        def hess_mult(point_array_coords, tangent_array_coords):
            return cfuncs.potential_energy_squared_distance_kernel_hess_mult(kernel_func, point_array_coords,
                                                                             tangent_array_coords)

        ManifoldObjectiveFunction.__init__(self, manifold, f, grad=grad, hess_mult=hess_mult,
                                           parameterized=parameterized)


class PotentialEnergyProjectiveDotProductKernel(ManifoldObjectiveFunction):
    def __init__(self, matrix_manifold, kernel_func, parameterized=False):
        def f(point_array_coords):
            return cfuncs.potential_energy_projective_dot_product_kernel_f(self, kernel_func, point_array_coords)

        def grad(point_array_coords):
            return cfuncs.potential_energy_projective_dot_product_kernel_grad(self, kernel_func, point_array_coords)

        def hess_mult(point_array_coords, tangent_array_coords):
            return cfuncs.potential_energy_projective_dot_product_kernel_hess_mult(self, kernel_func,
                                                                                   point_array_coords,
                                                                                   tangent_array_coords)

        ManifoldObjectiveFunction.__init__(self, matrix_manifold, f, grad, hess_mult, parameterized=parameterized)

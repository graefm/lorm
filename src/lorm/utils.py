import numpy as np
import copy as cp

def eval_objective_function_with_quadratic_approximation(func,tangent_vector_array,sample_size=100,interval=(-1,1)):
    func_values = np.zeros(sample_size)
    quad_values = np.zeros(sample_size)
    tv_sample = cp.deepcopy(tangent_vector_array)
    mp_sample = tv_sample.base_point_array
    f_x0 = func.f(mp_sample)
    f1_x0 = (func.grad(mp_sample).coords*tv_sample.coords).sum()
    f2_x0 = (func.hess_mult(tv_sample).coords*tv_sample.coords).sum()
    steps = np.zeros([sample_size])
    for i in range(sample_size):
        step_size = interval[0]+i*(1.*(interval[1]-interval[0]))/sample_size
        steps[i] = step_size
        tv_sample = cp.deepcopy(tangent_vector_array)
        mp_sample = tv_sample.base_point_array
        tv_sample.perform_geodesic_step(step_size)
        func_values[i] = func.f(mp_sample)
        quad_values[i] = f_x0 + f1_x0*step_size + 1./2.*f2_x0*(step_size**2)
    return func_values, quad_values, steps

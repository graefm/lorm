import copy as cp
from . import manif


class ArmijoLineSearch:

    def __init__(self, tol=1e-8, mu=0.4, tau=0.8, max_iter=12):
        self.tol = tol
        self.mu = mu
        self.tau = tau
        self.max_iter = max_iter

    def run(self, func, search_direction_vector_array, init_step_length=1.0):
        d0 = search_direction_vector_array
        p0 = d0.base_point_array
        f0 = func.f(p0)

        armijo_cond = self.mu * (func.grad(p0).coords * d0.coords).sum()
        alpha = init_step_length
        count = 0
        while True:
            d = cp.deepcopy(d0)
            d.perform_geodesic_step(alpha)
            if func.f(d.base_point_array) - f0 < alpha * armijo_cond:
                success = True
                break

            count += 1
            if count >= self.max_iter:
                d = cp.deepcopy(d0)
                success = False
                break

            alpha *= self.tau

        return d, success


class SteepestDescentMethod:

    def __init__(self, line_search_method=ArmijoLineSearch(), tol_grad=1e-8, max_iter=20, listener=None):
        self.ls = line_search_method
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.listener = listener

    def run(self, func, base_point_array, d_mask=None):
        p = base_point_array  # initial point
        f = func.f(p)  # initial function value
        if p.manifold.parameterized:
            d = manif.TangentVectorArrayParameterized(p)
        else:
            d = manif.TangentVectorArray(p)

        print("Initial value: f[0] = {val}".format(val=f))

        count = 0
        while True:
            # we shall compute new point by linesearch in direction of steepest descent
            g = func.grad(p)
            d.coords = -g.coords
            if d_mask is not None:
                d.coords *= d_mask

            dd = (d.coords ** 2).sum()
            if dd < self.tol_grad ** 2:
                print("Reached required tolerance of the gradient!")
                break

            # esitmate initial step length alpha by Hessian
            dH = func.hess_mult(d)
            dHd = (d.coords * dH.coords).sum()
            alpha = 1
            if dHd != 0:
                alpha = abs(dd / dHd)
            # run the line search
            d_new, ls_success = self.ls.run(func, d, init_step_length=alpha)
            if d_mask is not None:
                d_new.coords *= d_mask

            if ls_success:
                # set new iteration point
                p = d_new.base_point_array
                count += 1
                print("f[{i}] = {val}".format(i=count, val=func.f(p)))

                if count >= self.max_iter:
                    print("Reached maximal iterations!")
                    break
            else:
                print("Line search failed!")
                break

            if self.listener:
                self.listener(p)

            d.base_point_array.coords = p.coords

        return p


class ConjugateGradientMethod:

    def __init__(self, line_search_method=ArmijoLineSearch(), tol_grad=1e-8, max_iter=20, listener=None):
        self.ls = line_search_method
        self.tol_grad = tol_grad
        self.max_iter = max_iter
        self.listener = listener

    def run(self, func, base_point_array, d_mask=None):
        p = base_point_array
        f = func.f(p)
        g = func.grad(p)
        if p.manifold.parameterized:
            d = manif.TangentVectorArrayParameterized(p)
        else:
            d = manif.TangentVectorArray(p)
        d.coords = -g.coords
        if d_mask is not None:
            d.coords *= d_mask

        print("Initial value: f[0] = {val}".format(val=f))

        count = 0
        while True:
            # we shall compute new point by linesearch in direction d
            # and estimate the initial step length by Hessian
            dH = func.hess_mult(d)
            dHd = (d.coords * dH.coords).sum()
            alpha = 1
            if dHd != 0:
                gd = (g.coords * d.coords).sum()
                alpha = abs(gd / dHd)
            # run the line search
            d_new, ls_success = self.ls.run(func, d, init_step_length=alpha)
            if d_mask is not None:
                d_new.coords *= d_mask

            if ls_success:
                # set new iteration point
                p = d_new.base_point_array
                count += 1
                print("f[{i}] = {val}".format(i=count, val=func.f(p)))

                if count >= self.max_iter:
                    print("Reached maximal iterations!")
                    break
            else:
                print("Line search failed!")
                break
            # listener callback
            if self.listener:
                self.listener(p)

            g = func.grad(p)
            if d_mask is not None:
                g.coords *= d_mask
            gg = (g.coords ** 2).sum()
            if gg < self.tol_grad ** 2:
                print("Reached required tolerance of the gradient!")
                break

            d.base_point_array.coords = p.coords

            # compute new conjugate search direction
            Hd_new = func.hess_mult(d_new)
            dHd_new = (d_new.coords * Hd_new.coords).sum()
            if dHd_new != 0:
                gHd_new = (g.coords * Hd_new.coords).sum()
                beta = gHd_new / dHd_new
                d.coords = beta * d_new.coords - g.coords
                gd = (g.coords * d.coords).sum()
                if gd > 0:
                    d.coords = -g.coords
            else:
                d.coords = -g.coords

        return p

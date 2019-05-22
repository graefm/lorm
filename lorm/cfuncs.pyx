#!python

cimport cfuncs
import numpy as np
import math
from libc cimport math as cmath
cimport cython

cdef class KernelFunction:
    cpdef double f(self, double x) except *:
        return 0.
    cpdef double d1f(self, double x) except *:
        return 0.
    cpdef double d2f(self, double x) except *:
        return 0.

cdef class LogKernel(KernelFunction):
    cdef double scalar

    def __init__(self, scalar=-1.0):
            self.scalar = scalar

    cpdef double f(self, double x) except *:
            return self.scalar*cmath.log(x)
    @cython.cdivision(True)
    cpdef double d1f(self, double x) except *:
            return self.scalar/x
    @cython.cdivision(True)
    cpdef double d2f(self, double x) except *:
            return self.scalar*(-1.)/(x*x)


cdef class PowerKernel(KernelFunction):
    cdef double power
    cdef double a,b,c,d

    def __init__(self, power=1.0, a=1.0, b=1.0, c=0.0, d=0.0):
        # a*(b*x + c)**power + d
            self.power = power
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    cpdef double f(self, double x) except *:
            return self.a*cmath.pow(self.b*x+self.c,self.power) + self.d
    cpdef double d1f(self, double x) except *:
            return self.a*self.b*self.power*cmath.pow(self.b*x+self.c,self.power-1.0)
    cpdef double d2f(self, double x) except *:
            return self.a*self.b*self.b*self.power*(self.power-1.0)*cmath.pow(self.b*x+self.c,self.power-2.0)

cdef class PolynomialKernel(KernelFunction):
    cdef int degree
    cdef double[::1] coeffs
    cdef double[::1] d1coeffs
    cdef double[::1] d2coeffs

    def __init__(self, coeffs):
        self.coeffs = np.array(coeffs, dtype=np.float64)
        self.degree = self.coeffs.shape[0]-1
        self.d1coeffs = np.array(coeffs[1:self.degree+1], dtype=np.float64)*np.array(range(1,self.degree+1))
        self.d2coeffs = np.array(self.d1coeffs[1:self.degree], dtype=np.float64)*np.array(range(1,self.degree))
        # print(np.asarray(self.coeffs))
        # print(np.asarray(self.d1coeffs))
        # print(np.asarray(self.d2coeffs))

    cpdef double f(self, double x) except *:
        cdef double val = self.coeffs[self.degree]
        cdef int i
        if self.degree > 0:
            for i in range(1,self.degree+1):
                val *= x
                val += self.coeffs[self.degree-i]
        return val

    cpdef double d1f(self, double x) except *:
        cdef double val = 0
        cdef int i
        if self.degree > 0:
            val = self.d1coeffs[self.degree-1]
            if self.degree > 1:
                for i in range(1,self.degree):
                    val *= x
                    val += self.d1coeffs[self.degree-1-i]
        return val

    cpdef double d2f(self, double x) except *:
        cdef double val = 0
        cdef int i
        if self.degree > 1:
            val = self.d2coeffs[self.degree-2]
            if self.degree > 2:
                for i in range(1,self.degree-1):
                    val *= x
                    val += self.d2coeffs[self.degree-2-i]
        return val



def potential_energy_dot_product_kernel_f(KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords

    cdef double sum = 0
    cdef int i,j
    for i in range(M):
        sum += kernel.f(cfuncs.dot(x[i,:], x[i,:]))
        for j in range(i+1,M):
            sum += 2*kernel.f(cfuncs.dot(x[i,:], x[j,:]))
    return sum


def potential_energy_dot_product_kernel_grad(KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords
    grad = np.zeros_like(point_array_coords)
    cdef double[:,:] g = grad

    cdef double d1k_ij
    cdef int i,j,k
    for i in range(M):
        for j in range(i,M):
            d1k_ij = kernel.d1f(cfuncs.dot(x[i,:], x[j,:]))
            for k in range(dim):
                g[i,k] += 2.*d1k_ij*x[j,k]
            if j!=i:
                for k in range(dim):
                    g[j,k] += 2.*d1k_ij*x[i,k]
    return grad


def potential_energy_dot_product_kernel_hess_mult(KernelFunction kernel,point_array_coords,tangent_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords
    cdef double[:,::1] v = tangent_array_coords
    hess_mult_v = np.zeros_like(point_array_coords)
    cdef double[:,::1] mult_v = hess_mult_v

    cdef double d1k_ij, d2k_ij, s_ij, r_ij
    cdef int i,j,k
    for i in range(M):
        for j in range(i,M):
            s_ij = cfuncs.dot(x[i,:],x[j,:])
            d1k_ij = kernel.d1f(s_ij)
            d2k_ij = kernel.d2f(s_ij)
            r_ij = cfuncs.dot(x[i,:],v[j,:])+cfuncs.dot(x[j,:],v[i,:])
            for k in range(dim):
                mult_v[i,k] += 2.*d1k_ij*v[j,k]
                mult_v[i,k] += 2.*d2k_ij*x[j,k]*r_ij
            if i!=j:
                for k in range(dim):
                    mult_v[j,k] += 2.*d1k_ij*v[i,k]
                    mult_v[j,k] += 2.*d2k_ij*x[i,k]*r_ij
    return hess_mult_v


def potential_energy_squared_distance_kernel_f(KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords

    cdef double sum = 0, n2_ij
    cdef double[:] x_ij = np.empty(dim)
    cdef int i,j
    for i in range(M):
        for j in range(i+1,M):
            compute_diff(x[i,:], x[j,:], x_ij)
            n2_ij = cfuncs.dot(x_ij, x_ij)
            sum += 2*kernel.f(n2_ij)
    return sum


def potential_energy_squared_distance_kernel_grad(KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords
    grad = np.zeros_like(point_array_coords)
    cdef double[:,:] g = grad

    cdef double n2_ij, d1k_ij
    cdef double[:] x_ij = np.empty(dim)
    cdef int i,j,k
    for i in range(M):
        for j in range(i+1,M):
            compute_diff(x[i,:], x[j,:], x_ij)
            n2_ij = cfuncs.dot(x_ij,x_ij)
            d1k_ij = kernel.d1f(n2_ij)
            for k in range(dim):
                g[i,k] += 4.*d1k_ij*x_ij[k]
                g[j,k] -= 4.*d1k_ij*x_ij[k]
    return grad


def potential_energy_squared_distance_kernel_hess_mult(KernelFunction kernel,point_array_coords,tangent_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int dim = point_array_coords.shape[1]
    cdef double[:,::1] x = point_array_coords
    cdef double[:,::1] v = tangent_array_coords
    hess_mult_v = np.zeros_like(point_array_coords)
    cdef double[:,::1] mult_v = hess_mult_v

    cdef double n2_ij, d1k_ij, d2k_ij, x_ij_v_ij
    cdef double[:] x_ij = np.empty(dim), v_ij = np.empty(dim)
    cdef int i,j,k
    for i in range(M):
        for j in range(i+1,M):
            compute_diff(x[i,:], x[j,:], x_ij)
            compute_diff(v[i,:], v[j,:], v_ij)
            n2_ij = cfuncs.dot(x_ij,x_ij)
            d1k_ij = kernel.d1f(n2_ij)
            d2k_ij = kernel.d2f(n2_ij)
            x_ij_v_ij = cfuncs.dot(x_ij,v_ij)
            for k in range(dim):
                mult_v[i,k] += 4.*d1k_ij*v_ij[k]
                mult_v[j,k] -= 4.*d1k_ij*v_ij[k]
                mult_v[i,k] += 8.*d2k_ij*x_ij[k]*x_ij_v_ij
                mult_v[j,k] -= 8.*d2k_ij*x_ij[k]*x_ij_v_ij
    return hess_mult_v

def potential_energy_projective_dot_product_kernel_f(self,KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int rows = self.manifold.matrix_size[0]
    cdef int cols = self.manifold.matrix_size[1]
    cdef double [:,:,::1] x = np.asarray(point_array_coords.reshape((M,rows,cols)))
    cdef double [:,::1] x_ij = np.empty((cols,cols))
    cdef double sum=0, n2_ij, k_ij

    cdef int i,j
    for i in range(M):
        for j in range(i+1,M):
                cfuncs.matrix_mult_atb(x[i,:,:],x[j,:,:],x_ij)
                n2_ij = cfuncs.trace_inner_product(x_ij,x_ij)
                k_ij = kernel.f(n2_ij)
                sum += 2*k_ij

    return sum

def potential_energy_projective_dot_product_kernel_grad(self,KernelFunction kernel,point_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int rows = self.manifold.matrix_size[0]
    cdef int cols = self.manifold.matrix_size[1]
    cdef double [:,:,::1] x = np.asarray(point_array_coords.reshape((M,rows,cols)))
    cdef double [:,::1] x_ij = np.empty((cols,cols))
    grad = np.zeros_like(point_array_coords.reshape((M,rows,cols)))
    cdef double [:,:,::1] g = grad

    cdef double d1k_ij, n2_ij

    cdef int i,j,k,l
    for i in range(M):
        for j in range(i+1,M):
                cfuncs.matrix_mult_atb(x[i,:,:],x[j,:,:],x_ij)
                n2_ij = cfuncs.trace_inner_product(x_ij,x_ij)
                d1k_ij = kernel.d1f(n2_ij)
                #grad[i,:,:] += 4*np.dot(x[j,:,:],np.dot(x[j,:,:].T,x[i,:,:]))*d1k_ij
                #grad[j,:,:] += 4*np.dot(x[i,:,:],np.dot(x[i,:,:].T,x[j,:,:]))*d1k_ij
                for k in range(rows):
                    for l in range(cols):
                        g[i,k,l] += 4*d1k_ij*cfuncs.dot(x[j,k,:],x_ij[l,:])
                        g[j,k,l] += 4*d1k_ij*cfuncs.dot(x[i,k,:],x_ij[:,l])

    return grad.reshape((M,rows*cols))

def potential_energy_projective_dot_product_kernel_hess_mult(self,KernelFunction kernel,point_array_coords,tangent_array_coords):
    cdef int M = point_array_coords.shape[0]
    cdef int rows = self.manifold.matrix_size[0]
    cdef int cols = self.manifold.matrix_size[1]
    cdef double [:,:,::1] x = np.asarray(point_array_coords.reshape((M,rows,cols)))
    cdef double [:,::1] x_ij = np.empty((cols,cols))
    cdef double [:,::1] xv_ji = np.empty((cols,cols))
    cdef double [:,::1] vx_ji = np.empty((cols,cols))
    cdef double [:,::1] t1 = np.empty((rows,cols))
    cdef double [:,::1] t2 = np.empty((rows,cols))
    cdef double [:,::1] t3 = np.empty((rows,cols))
    cdef double [:,:,::1] v = np.asarray(tangent_array_coords.reshape((M,rows,cols)))
    hess_mult_v = np.zeros_like(point_array_coords.reshape((M,rows,cols)))
    cdef double [:,:,::1] hv = hess_mult_v

    cdef double n2_ij, d1k_ij, d2k_ij, t1vi, t2vj

    cdef int i,j
    for i in range(M):
        for j in range(M):
            if j!=i:
                #xij = np.dot(x[i,:,:].T,x[j,:,:])
                cfuncs.matrix_mult_atb(x[i,:,:],x[j,:,:],x_ij)
                cfuncs.matrix_mult_atb(x[j,:,:],v[i,:,:],xv_ji)
                cfuncs.matrix_mult_atb(v[j,:,:],x[i,:,:],vx_ji)
                n2_ij = cfuncs.trace_inner_product(x_ij,x_ij)
                d1k_ij = kernel.d1f(n2_ij)
                d2k_ij = kernel.d2f(n2_ij)
                #temp1 = np.dot(x[j,:,:],np.dot(x[j,:,:].T,x[i,:,:]))
                #temp2 = np.dot(x[i,:,:],np.dot(x[i,:,:].T,x[j,:,:]))
                for k in range(rows):
                    for l in range(cols):
                        t1[k,l] = cfuncs.dot(x[j,k,:],x_ij[l,:])
                        t2[k,l] = cfuncs.dot(x[i,k,:],x_ij[:,l])
                #temp3  = np.dot(x[j,:,:],np.dot(x[j,:,:].T,v[i,:,:]))
                #temp3 += np.dot(v[j,:,:],np.dot(x[j,:,:].T,x[i,:,:]))
                #temp3 += np.dot(x[j,:,:],np.dot(v[j,:,:].T,x[i,:,:]))
                for k in range(rows):
                    for l in range(cols):
                        t3[k,l] = cfuncs.dot(x[j,k,:],xv_ji[:,l])
                for k in range(rows):
                    for l in range(cols):
                        t3[k,l] += cfuncs.dot(v[j,k,:],x_ij[l,:])
                for k in range(rows):
                    for l in range(cols):
                        t3[k,l] += cfuncs.dot(x[j,k,:],vx_ji[:,l])

                #hess_mult_v[i,:,:] += 8*temp1*d2k_ij*np.sum(temp1*v[i,:,:])
                #hess_mult_v[i,:,:] += 8*temp1*d2k_ij*np.sum(temp2*v[j,:,:])
                #hess_mult_v[i,:,:] += 4*temp3*d1k_ij
                t1vi = cfuncs.trace_inner_product(t1,v[i,:,:])
                t2vj = cfuncs.trace_inner_product(t2,v[j,:,:])
                for k in range(rows):
                    for l in range(cols):
                        hv[i,k,l] += 8*t1[k,l]*d2k_ij*t1vi
                        hv[i,k,l] += 8*t1[k,l]*d2k_ij*t2vj
                        hv[i,k,l] += 4*t3[k,l]*d1k_ij


    return hess_mult_v.reshape((M,rows*cols))

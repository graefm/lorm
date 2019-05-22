cdef class KernelFunction:
    cpdef double f(self, double x) except *
    cpdef double d1f(self, double x) except *
    cpdef double d2f(self, double x) except *

cdef inline double dot(double[:] x, double[:] y):
    cdef int dim = x.shape[0]
    cdef int k
    cdef double s = 0
    for k in range(dim):
        s += x[k]*y[k]
    return s

cdef inline int compute_diff(double[:] a, double[:] b, double[:] result):
    cdef int dim = a.shape[0]
    cdef int k
    for k in range(dim):
        result[k] = a[k] - b[k]
    return 0

cdef inline double trace_inner_product(double [:,::1] a, double [:,::1] b ):
    cdef int rows = a.shape[0]
    cdef int cols = a.shape[1]
    cdef double sum = 0
    for k in range(rows):
        sum += dot(a[k,:],b[k,:])
    return sum

cdef inline matrix_mult_atb(double [:,::1] a, double [:,::1] b, double [:,::1] atb):
    cdef int rows = a.shape[0]
    cdef int cols = a.shape[1]
    for k in range(cols):
        for l in range(cols):
            atb[k,l] = dot(a[:,k],b[:,l])
    return    

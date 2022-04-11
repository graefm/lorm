from cnfft cimport *
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
np.import_array()

# initialize FFTW threads
#fftw_init_threads()
#fftw_plan_with_nthreads(1)

# register cleanup callbacks
#cdef void _cleanup():
#    fftw_cleanup()
#    fftw_cleanup_threads()

#Py_AtExit(_cleanup)

cdef max(int a, int b):
  if a >= b:
    return a
  else:
    return b

def _set_fhat(self,fhat):
  cdef int N = fhat.N, N_internal =self._N_internal
  assert N <= N_internal
  cdef complex[:] f_hat = self._cplan.f_hat
  cdef int i=0,k1,k2,n,k1k2max,
  for k2 in range(-N_internal,N_internal+1):
    for k1 in range(-N_internal,N_internal+1):
      k1k2max = max(abs(k1),abs(k2))
      for n in range(k1k2max,N_internal+1):
        f_hat[i] = 0
        if n <= N:
          f_hat[i] = fhat.array[(2*n-1)*(n)*(2*n+1)/3+(2*n+1)*(n+k1)+(n+k2)]#fhat[n,k1,k2]
        i += 1

def _get_fhat(self,fhat):
  cdef int N = fhat.N, N_internal =self._N_internal
  cdef complex[:] f_hat = self._cplan.f_hat
  cdef int i=0,k1,k2,n,k1k2max,
  for k2 in range(-N_internal,N_internal+1):
    for k1 in range(-N_internal,N_internal+1):
      k1k2max = max(abs(k1),abs(k2))
      for n in range(k1k2max,N_internal+1):
        if n <= N:
          fhat.array[(2*n-1)*(n)*(2*n+1)/3+(2*n+1)*(n+k1)+(n+k2)] = f_hat[i] #fhat[n,k1,k2]
        i += 1



cdef class plan:
  cdef nfsoft_plan _plan
  cdef object _f_hat
  cdef object _f
  cdef object _x

  def __cinit__(self):
    pass

  def __init__(self, M, N, m=7):
    '''
    M number of points
    N bandwith
    m nfft cutoff parameter
    '''
    # initialize nfsft plan
    cdef unsigned nfft_flags = PRE_PHI_HUT | PRE_PSI | FFTW_INIT | MALLOC_X | MALLOC_F_HAT | MALLOC_F | FFT_OUT_OF_PLACE | NFFT_OMP_BLOCKWISE_ADJOINT # | NFFT_SORT_NODES
    cdef unsigned nfsoft_flags = NFSOFT_MALLOC_F | NFSOFT_MALLOC_F_HAT | NFSOFT_MALLOC_X
    nfsoft_init_guru( &self._plan, N, M, nfsoft_flags, nfft_flags, m, 1000)

    # initialize memory views
    cdef np.npy_intp _f_hat_dims[1]
    _f_hat_dims[0] = (2*N+1)*(2*N+2)*(2*N+3)/6
    self._f_hat = np.PyArray_SimpleNewFromData(1, _f_hat_dims, np.NPY_COMPLEX128, <void *>(self._plan.f_hat))

    cdef np.npy_intp _f_dims[1]
    _f_dims[0] = M
    self._f = np.PyArray_SimpleNewFromData(1, _f_dims, np.NPY_COMPLEX128, <void *>(self._plan.f))

    cdef np.npy_intp _x_dims[2]
    _x_dims[0] = M
    _x_dims[1] = 3
    self._x = np.PyArray_SimpleNewFromData(2, _x_dims, np.NPY_FLOAT64, <void *>(self._plan.x))

    self._f_hat[:] = 0
    self._f[:] = 0
    self._x[:] = 0
    pass

  def __dealloc__(self):
      nfsoft_finalize(&self._plan)

  def precompute_x(self):
    nfsoft_precompute( &self._plan )

  def trafo(self):
    nfsoft_trafo(&self._plan)
    return

  def adjoint(self):
    nfsoft_adjoint(&self._plan)
    return

  @property
  def f_hat(self):
    return self._f_hat
  @f_hat.setter
  def f_hat(self, array):
      self._f_hat.ravel()[:] = array.ravel()

  @property
  def f(self):
    return self._f
  @f.setter
  def f(self, array):
      self._f.ravel()[:] = array.ravel()

  @property
  def x(self):
    return self._x
  @x.setter
  def x(self, array):
    self._x.ravel()[:] = array.ravel()

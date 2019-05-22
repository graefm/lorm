from cnfft cimport *
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
np.import_array()

# initialize FFTW threads
#fftw_init_threads()

# register cleanup callbacks
#cdef void _cleanup():
#    fftw_cleanup()
#    fftw_cleanup_threads()

#Py_AtExit(_cleanup)

cdef class plan:
  cdef int[:] _N
  cdef int[:] _n
  cdef nfft_plan _plan
  cdef object _f_hat
  cdef object _f
  cdef object _x

  def __cinit__(self):
    pass

  def __init__(self, M, N, n, m):
    '''
    M number of points
    N tuple of bandwith
    n tuple of internal bandwith
    m cutoff parameter
    '''
    # initialize nfft plan
    cdef int d = len(N)
    self._N = np.array(N,dtype=np.int32)
    self._n = np.array(n,dtype=np.int32)
    cdef unsigned flags = PRE_PHI_HUT | PRE_PSI | MALLOC_X | MALLOC_F_HAT | MALLOC_F | FFTW_INIT | FFT_OUT_OF_PLACE | NFFT_OMP_BLOCKWISE_ADJOINT | NFFT_SORT_NODES

    cdef unsigned fftw_flags = FFTW_ESTIMATE | FFTW_DESTROY_INPUT
    nfft_init_guru(&self._plan, d, &self._N[0], M, &self._n[0], m, flags, fftw_flags)

    # initialize memory views
    cdef np.npy_intp * _f_hat_dims = <np.npy_intp *> malloc(d * sizeof(np.npy_intp))
    for i in range(d):
      _f_hat_dims[i] = self._N[i]
    self._f_hat = np.PyArray_SimpleNewFromData(d, _f_hat_dims, np.NPY_COMPLEX128, <void *>(self._plan.f_hat))
    free(_f_hat_dims)

    cdef np.npy_intp _f_dims[1]
    _f_dims[0] = M
    self._f = np.PyArray_SimpleNewFromData(1, _f_dims, np.NPY_COMPLEX128, <void *>(self._plan.f))

    cdef np.npy_intp _x_dims[2]
    _x_dims[0] = M
    _x_dims[1] = d
    self._x = np.PyArray_SimpleNewFromData(2, _x_dims, np.NPY_FLOAT64, <void *>(self._plan.x))

    self._f_hat[:] = 0
    self._f[:] = 0
    self._x[:] = 0
    pass

  def __dealloc__(self):
      nfft_finalize(&self._plan)

  def precompute_x(self):
    if self._plan.flags & PRE_ONE_PSI:
      nfft_precompute_one_psi(&self._plan)

  def trafo(self, direct=False):
    if direct:
      nfft_trafo_direct(&self._plan)
    else:
      nfft_trafo(&self._plan)
    return

  def adjoint(self, direct=False):
    if direct:
      nfft_adjoint_direct(&self._plan)
    else:
      nfft_adjoint(&self._plan)
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

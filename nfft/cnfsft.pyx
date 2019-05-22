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
  cdef nfsft_plan _plan
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
    cdef unsigned nfft_flags = PRE_PHI_HUT | PRE_PSI | FFTW_INIT | FFT_OUT_OF_PLACE | NFFT_OMP_BLOCKWISE_ADJOINT #| NFFT_SORT_NODES 
    cdef unsigned nfsft_flags = NFSFT_MALLOC_F | NFSFT_MALLOC_F_HAT | NFSFT_MALLOC_X | NFSFT_NORMALIZED
    nfsft_precompute( N, 1000.0, 0U, 0U);
    nfsft_init_guru(&self._plan, N, M, nfsft_flags, nfft_flags, m)

    # initialize memory views
    cdef np.npy_intp _f_hat_dims[2]
    _f_hat_dims[0] = 2*N+2
    _f_hat_dims[1] = 2*N+2
    self._f_hat = np.PyArray_SimpleNewFromData(2, _f_hat_dims, np.NPY_COMPLEX128, <void *>(self._plan.f_hat))

    cdef np.npy_intp _f_dims[1]
    _f_dims[0] = M
    self._f = np.PyArray_SimpleNewFromData(1, _f_dims, np.NPY_COMPLEX128, <void *>(self._plan.f))

    cdef np.npy_intp _x_dims[2]
    _x_dims[0] = M
    _x_dims[1] = 2
    self._x = np.PyArray_SimpleNewFromData(2, _x_dims, np.NPY_FLOAT64, <void *>(self._plan.x))

    self._f_hat[:] = 0
    self._f[:] = 0
    self._x[:] = 0
    pass

  def __dealloc__(self):
      nfsft_finalize(&self._plan)

  def trafo(self, direct=False):
    if direct:
      nfsft_trafo_direct(&self._plan)
    else:
      nfsft_trafo(&self._plan)
    return

  def adjoint(self, direct=False):
    if direct:
      nfsft_adjoint_direct(&self._plan)
    else:
      nfsft_adjoint(&self._plan)
    return

  def precompute_x(self):
    nfsft_precompute_x(&self._plan)

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

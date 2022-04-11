import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
np.import_array()

cpdef linearized_index(int M0, int M1, int K0, int K1):
    if M0 < M1:
        return K1+M1+(-1+K0-M0)*(1+2*M1)+M1**4+(1+2*M1)*(1-M0**2+2*M1*(1+M1))
    else:
        return K1+M1+(-1+K0-M0)*(1+2*M1)+M0**4+(1+2*M0)*(1+M1)**2


def _set_f_hat_internal(self,f_hat):
  cdef int N = f_hat.N, N_internal =self._N_internal
  assert N <= N_internal
  cdef complex[:] f_hat_internal = self._f_hat_internal
  cdef int i=0,j=0,k0,k1,m0,m1
  for k0 in range(-N_internal,N_internal+1):
    for k1 in range(-N_internal,N_internal+1):
      for m0 in range(abs(k0),N_internal+1):
          for m1 in range(abs(k1),N_internal+1):
              f_hat_internal[i] = 0
              if m0 <= N and m1 <= N:
                  f_hat_internal[i] = f_hat.array[linearized_index(m0,m1,k0,k1)]
                  #j+=1
              i+=1

def _get_from_f_hat_internal(self,f_hat):
  cdef int N = f_hat.N, N_internal =self._N_internal
  cdef complex[:] f_hat_internal = self._f_hat_internal
  cdef int i=0,j=0,k0,k1,m0,m1
  for k0 in range(-N_internal,N_internal+1):
    for k1 in range(-N_internal,N_internal+1):
        for m0 in range(abs(k0),N_internal+1):
            for m1 in range(abs(k1),N_internal+1):
                if m0 <= N and m1 <= N:
                    f_hat.array[linearized_index(m0,m1,k0,k1)] = f_hat_internal[i]
                    #j+=1
                i += 1

#!python
#cython: language_level=3, boundscheck=False, wraparound=False
from libc.math cimport sin,cos,acos,atan,sqrt,M_PI,atan2
import numpy as np

def rotation_group_christoffel_matrices(self, base_point_coords):
        cdef int dim_m = self._dim_matrix
        cdef int dim = self._dim
        cdef double[:,:] r = self.coords_as_matrix(base_point_coords)
        christoffel_matrices = np.zeros((dim_m,)*6)
        cdef double[:,:,:,:,:,:] cm = christoffel_matrices

        cdef int i,j,k,l,m,n
        for i in range(dim_m):
            for j in range(dim_m):
                for k in range(dim_m):
                    for l in range(dim_m):
                        cm[i,j,i,k,l,j] += r[l,k]
                        cm[i,k,i,j,l,j] -= r[l,k]
                        cm[k,j,i,l,i,j] -= r[k,l]
                        for m in range(dim_m):
                            for n in range(dim_m):
                                cm[i,j,k,l,m,n] += r[i,l]*r[m,j]*r[k,n]

        christoffel_matrices.shape = (dim,dim,dim)
        for i in range(dim):
              christoffel_matrices[i,:,:] += christoffel_matrices[i,:,:].transpose()

        christoffel_matrices /= -8.

        return christoffel_matrices

def rotation_group_christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
    cdef int dim_m = self._matrix_size[0]
    cdef int dim = self._dim
    r_m = self.coords_as_matrix(base_point_coords)
    c_m = self.coords_as_matrix(coeffs)
    rc_m = r_m.transpose()*c_m
    cr_m = c_m * r_m.transpose()
    rcr_m = rc_m * r_m.transpose()
    matrix = np.zeros((dim_m,)*4)
    cdef double[:,:,:,:] cm = matrix
    cdef double[:,:] r = r_m
    cdef double[:,:] c = c_m
    cdef double[:,:] rc = rc_m
    cdef double[:,:] cr = cr_m
    cdef double[:,:] rcr = rcr_m

    cdef int k,l,m,n
    for k in range(dim_m):
        for l in range(dim_m):
            for m in range(dim_m):
                cm[k,l,m,l] -= cr[k,m] + cr[m,k]
                cm[k,l,k,m] -= rc[l,m] + rc[m,l]
                for n in range(dim_m):
                    cm[k,l,m,n] += rcr[l,m]*r[k,n] + c[k,n]*r[m,l]
                    cm[k,l,m,n] += rcr[n,k]*r[m,l] + c[m,l]*r[k,n]

    matrix.shape = (dim,dim)
    matrix /= -8.

    return matrix

def SO3_parameterization(double phi1, double theta, double phi2):
    cdef double sp1 = sin(phi1), cp1 = cos(phi1), st = sin(theta), ct = cos(theta), sp2 = sin(phi2), cp2 = cos(phi2)
    matrix_np = np.empty([3,3])
    cdef double[:,:] matrix = matrix_np
    matrix[0,0] = cp1 * ct * cp2 - sp1 * sp2
    matrix[1,0] = sp1 * ct * cp2 + cp1 * sp2
    matrix[2,0] =  - st  * cp2
    matrix[0,1] =  - cp1 * ct * sp2 - sp1  * cp2
    matrix[1,1] =  - sp1 * ct * sp2 + cp1  * cp2
    matrix[2,1] =    st  * sp2
    matrix[0,2] =    cp1 * st
    matrix[1,2] =    sp1 * st
    matrix[2,2] =    ct
    return matrix_np.reshape(9)

def SO3_inverse_parameterization(double[:,:,:] matrix):
  # theta = arccos( m33 );
  local_coords_np = np.empty([matrix.shape[0],3])
  cdef double[:,:] local_coords = local_coords_np
  cdef double theta, phi1, phi2

  for i in range(matrix.shape[0]):
      theta = acos(matrix[i,2,2])
      # phi1 = 2 arctan( m23/ ( m13-sqrt(m13^2+m23^2)) ) + pi;
      # phi2 = 2 arctan( m32/ (-m31-sqrt(m31^2+m32^2)) ) + pi;
      phi1 = atan2(matrix[i,1,2],matrix[i,0,2])
      phi2 = atan2(matrix[i,2,1],-matrix[i,2,0])
      if phi1 < 0:
          phi1 += 2*M_PI
      if phi2 < 0:
          phi2 += 2*M_PI
      local_coords[i,0] = phi1
      local_coords[i,1] = theta
      local_coords[i,2] = phi2
  return local_coords_np
  #cdef double phi1 = matrix[1,2] / (   matrix[0,2] - sqrt( matrix[0,2]*matrix[0,2] + matrix[1,2]*matrix[1,2] ) )
  #cdef double phi2 = matrix[2,1] / ( - matrix[2,0] - sqrt( matrix[2,0]*matrix[2,0] + matrix[2,1]*matrix[2,1] ) )
  #phi1 = 2 * atan( phi1 ) + M_PI
  #phi2 = 2 * atan( phi2 ) + M_PI
  #if matrix[1,2] == 0 and matrix[2,1] == 0 and matrix[0,2] == 0 and matrix[2,0] == 0:
#    phi2 = 0
#    phi1 = acos( matrix[1,1] )
#    return np.array((phi1,theta,phi2))
 # if matrix[1,2] == 0 and 0 <= matrix[0,2]:
#    phi1 = 0
 # if matrix[2,1] == 0 and matrix[2,0] <= 0:
#    phi2 = 0
#  return np.array((phi1,theta,phi2))


def SO3_jacobi_matrix(double[:] phi1, double[:] theta, double[:] phi2):
    jacobi_matrix_np = np.empty([theta.shape[0],3,3,3])
    cdef double[:,:,:,:] jacobi_matrix = jacobi_matrix_np
    cdef double sp1, cp1, st, ct, sp2, cp2

    for i in range(theta.shape[0]):
        sp1 = sin(phi1[i])
        cp1 = cos(phi1[i])
        st = sin(theta[i])
        ct = cos(theta[i])
        sp2 = sin(phi2[i])
        cp2 = cos(phi2[i])
        # dphi1
        jacobi_matrix[i,0,0,0] =  - sp1  * ct * cp2 - cp1 * sp2
        jacobi_matrix[i,1,0,0] =    cp1  * ct * cp2 - sp1 * sp2
        jacobi_matrix[i,2,0,0] =    0
        jacobi_matrix[i,0,1,0] =    sp1  * ct * sp2 - cp1  * cp2
        jacobi_matrix[i,1,1,0] =  - cp1  * ct * sp2 - sp1  * cp2
        jacobi_matrix[i,2,1,0] =    0
        jacobi_matrix[i,0,2,0] =  - sp1  * st
        jacobi_matrix[i,1,2,0] =    cp1  * st
        jacobi_matrix[i,2,2,0] =    0
        # dtheta
        jacobi_matrix[i,0,0,1] =  - cp1 * st * cp2
        jacobi_matrix[i,1,0,1] =  - sp1 * st * cp2
        jacobi_matrix[i,2,0,1] =  - ct  * cp2
        jacobi_matrix[i,0,1,1] =    cp1 * st * sp2
        jacobi_matrix[i,1,1,1] =    sp1 * st * sp2
        jacobi_matrix[i,2,1,1] =    ct  * sp2
        jacobi_matrix[i,0,2,1] =    cp1 * ct
        jacobi_matrix[i,1,2,1] =    sp1 * ct
        jacobi_matrix[i,2,2,1] =  - st
        # dphi2
        jacobi_matrix[i,0,0,2] =  - cp1  * ct * sp2 - sp1 * cp2
        jacobi_matrix[i,1,0,2] =  - sp1  * ct * sp2 + cp1 * cp2
        jacobi_matrix[i,2,0,2] =    st * sp2
        jacobi_matrix[i,0,1,2] =  - cp1  * ct * cp2 + sp1 * sp2
        jacobi_matrix[i,1,1,2] =  - sp1  * ct * cp2 - cp1 * sp2
        jacobi_matrix[i,2,1,2] =    st * cp2
        jacobi_matrix[i,0,2,2] =    0
        jacobi_matrix[i,1,2,2] =    0
        jacobi_matrix[i,2,2,2] =    0
    return jacobi_matrix_np.reshape(theta.shape[0],9,3)

def SO3_inverse_riemannian_matrix(double[:] theta):
    cdef double[:,:,:] inverse_riemannian_matrix = np.empty([theta.shape[0],3,3])
    cdef double one_overst2, ct
    for i in range(theta.shape[0]):
        one_overst2 = 1./(sin(theta[i])*sin(theta[i]))
        ct = cos(theta[i])
        # phi1 column
        inverse_riemannian_matrix[i,0,0] = 0.5 * one_overst2
        inverse_riemannian_matrix[i,1,0] = 0
        inverse_riemannian_matrix[i,2,0] = - 0.5 * ct * one_overst2
        # theta column
        inverse_riemannian_matrix[i,0,1] = 0
        inverse_riemannian_matrix[i,1,1] = 0.5
        inverse_riemannian_matrix[i,2,1] = 0
        # phi2 column
        inverse_riemannian_matrix[i,0,2] = - 0.5 * ct * one_overst2
        inverse_riemannian_matrix[i,1,2] = 0
        inverse_riemannian_matrix[i,2,2] = 0.5 * one_overst2
    return inverse_riemannian_matrix

def SO3_christoffel_matrix_lin_comb_parameterization(double theta, double coef_phi1, double coef_theta, double coef_phi2):
    cdef st = sin(theta), over_st = 1./st, ct_over_st = cos(theta)/st
    cdef double[:,:] christoffel_matrix = np.zeros([3,3])
    # phi1 column
    christoffel_matrix[1,0] = 0.5 * (  coef_phi1 * ct_over_st - coef_phi2 * over_st )
    christoffel_matrix[2,0] = 0.5 * (  coef_theta * st )
    # theta column
    christoffel_matrix[0,1] = 0.5 * (  coef_phi1 * ct_over_st - coef_phi2 * over_st )
    christoffel_matrix[2,1] = 0.5 * ( -coef_phi1 * over_st + coef_phi2 * ct_over_st )
    # phi2 column
    christoffel_matrix[0,2] = 0.5 * (  coef_theta * st )
    christoffel_matrix[1,2] = 0.5 * ( -coef_phi1 * over_st + coef_phi2 * ct_over_st )
    return christoffel_matrix


def grassmannian_stiefel_rep_christoffel_matrices(self, base_point_coords):
    cdef int dim = self._dim
    cdef int dim_d = self._matrix_size[0]
    cdef int dim_k = self._matrix_size[1]
    matrix = np.zeros((dim_d,dim_k)*3)
    cdef double [:,:,:,:,:,:] cm = matrix

    p_m = self.coords_as_matrix(base_point_coords)
    cdef double[:,:] p = p_m
    cdef double[:,:] id_ppt_2 = (np.eye(dim_d)-p_m*p_m.transpose())/2.
    cdef double temp
    cdef int i,j,k,l,m,n
    for i in range(dim_d):
        for j in range(dim_k):
            for k in range(dim_d):
                for l in range(dim_k):
                    for m in range(dim_d):
                        temp = p[i,l]*id_ppt_2[k,m]
                        cm[i,j,k,l,m,j] += temp
                        cm[i,j,k,j,m,l] += temp

    matrix.shape = (dim,dim,dim)
    return matrix

def grassmannian_stiefel_rep_christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
    cdef int dim = self._dim
    cdef int dim_d = self._matrix_size[0]
    cdef int dim_k = self._matrix_size[1]
    matrix = np.empty((dim_d,dim_k)*2)
    cdef double[:,:,:,:] cm = matrix
    c_m = self.coords_as_matrix(coeffs)
    p_m = self.coords_as_matrix(base_point_coords)
    cdef double[:,:] id_ppt = (np.eye(dim_d)-p_m*p_m.transpose())
    cdef double[:,:] ptc_ctp_2 = (p_m.transpose()*c_m + c_m.transpose()*p_m)/2.

    cdef int i,j,k,l
    for i in range(dim_d):
        for j in range(dim_k):
            for k in range(dim_d):
                for l in range(dim_k):
                    cm[i,j,k,l] = id_ppt[i,k]*ptc_ctp_2[j,l]

    matrix.shape = (dim,dim)
    return matrix

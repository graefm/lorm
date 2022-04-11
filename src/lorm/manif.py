import numpy as np
import copy as cp
from scipy.linalg import expm
from . import cmanif


class ManifoldPointArray:
    def __init__(self, manifold):
        self._manifold = cp.deepcopy(manifold)
        self._coords = np.array([])

    def __str__(self):
        return "Array of {num} points of the manifold: ".format(num=len(self._coords))+ str(self._manifold)

    @property
    def manifold(self):
        return self._manifold

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = self._manifold.project_on_manifold(coords)


class ManifoldPointArrayParameterized(ManifoldPointArray):
    def __init__(self, manifold):
        assert manifold.parameterized
        self._local_coords = np.array([])
        ManifoldPointArray.__init__(self,manifold)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        ManifoldPointArray.coords.fset(self,coords)
        self._local_coords = self._manifold.compute_inverse_parameterization(self._coords)
        #self._local_coords = np.empty([coords.shape[0],self._manifold.local_dim])
        #inverse_parameterization = self._manifold.compute_inverse_parameterization
        #for i, point in enumerate(self._coords):
        #    self._local_coords[i] = inverse_parameterization(point)

    @property
    def local_coords(self):
        return self._local_coords

    @local_coords.setter
    def local_coords(self, local_coords):
        self._local_coords = np.empty(local_coords.shape)
        self._local_coords[:] = local_coords
        self._coords = self._manifold.compute_parameterization(local_coords)

class TangentVectorArray:
    def __init__(self, manifold_point_array):
        self._base_point_array = cp.deepcopy(manifold_point_array)
        self._coords = np.zeros(self._base_point_array.coords.shape)

    def __str__(self):
        return "Array of {num} tangent vectors of the manifold: ".format(num=len(self._coords)) \
          + str(self._base_point_array.manifold)

    @property
    def base_point_array(self):
        return self._base_point_array

    @property
    def manifold(self):
        return self._base_point_array.manifold

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = self.manifold.project_on_tangent_space(self._base_point_array.coords,coords)

    def perform_geodesic_step(self, step_length=1):
        self._base_point_array._coords, self._coords = self.manifold.geodesic_step(self._base_point_array.coords, self.coords, step=step_length)

    def normal_vector_coords(self):
        return self.manifold.normal_vector(self._base_point_array._coords,self._coords)

    def christoffel_matrix_lin_comb_mult(self, coeffs):
        christoffel_lin_comb = self.manifold.christoffel_matrix_lin_comb
        base_coords = self._base_point_array._coords
        mult_coords = np.empty(self._coords.shape)
        for i, tangent_coords in enumerate(self._coords):
            matrix = christoffel_lin_comb(base_coords[i], coeffs[i])
            mult_coords[i] = np.dot(matrix, tangent_coords)
        return mult_coords


class TangentVectorArrayParameterized(TangentVectorArray):
    def __init__(self, manifold_point_array):
        assert manifold_point_array.manifold.parameterized
        TangentVectorArray.__init__(self,manifold_point_array)
        self._local_coords = np.zeros(self._base_point_array.local_coords.shape)

    def perform_geodesic_step(self, step_length=1):
        TangentVectorArray.perform_geodesic_step(self, step_length)
        self._base_point_array._local_coords = self.manifold.compute_inverse_parameterization(self._base_point_array._coords)
        jacobi_matrix = self.manifold.compute_jacobi_matrix(self._base_point_array._local_coords)
        inverse_riemannian_matrix = self.manifold.compute_inverse_riemannian_matrix(self._base_point_array._local_coords)
        jacobi_transp_dot_coords = np.zeros([self._coords.shape[0],self.manifold.local_dim])
        for i in range(self.manifold.local_dim):
            for j in range(self.manifold.ambient_dim):
                jacobi_transp_dot_coords[:,i] += jacobi_matrix[:,j,i]*self._coords[:,j]
        self._local_coords = np.zeros([self._coords.shape[0],self.manifold.local_dim])
        for i in range(self.manifold.local_dim):
            for j in range(self.manifold.local_dim):
                self._local_coords[:,i] += inverse_riemannian_matrix[:,i,j] * jacobi_transp_dot_coords[:,j]

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        TangentVectorArray.coords.fset(self,coords)
        jacobi_matrix = self.manifold.compute_jacobi_matrix(self._base_point_array._local_coords)
        inverse_riemannian_matrix = self.manifold.compute_inverse_riemannian_matrix(self._base_point_array._local_coords)
        jacobi_transp_dot_coords = np.zeros([coords.shape[0],self.manifold.local_dim])
        for i in range(self.manifold.local_dim):
            for j in range(self.manifold.ambient_dim):
                jacobi_transp_dot_coords[:,i] += jacobi_matrix[:,j,i]*coords[:,j]
        self._local_coords = np.zeros([coords.shape[0],self.manifold.local_dim])
        for i in range(self.manifold.local_dim):
            for j in range(self.manifold.local_dim):
                self._local_coords[:,i] += inverse_riemannian_matrix[:,i,j] * jacobi_transp_dot_coords[:,j]

    @property
    def local_coords(self):
        return self._local_coords

    @local_coords.setter
    def local_coords(self, local_coords):
        self._local_coords = local_coords
        jacobi_matrix = self.manifold.compute_jacobi_matrix(self._base_point_array._local_coords)
        for i, base_point in enumerate(self._base_point_array.local_coords):
            self._coords[i] = np.dot(jacobi_matrix[i,:,:], local_coords[i])

class Manifold:
    def __init__(self):
        # set to True if a parameterization is implemented, (e.g. see Shpere2)
        self._parameterized = False

    def __str__(self):
        return self._description

    def __eq__(self, other):
        if isinstance(other, Manifold):
            return self._description == other._description
        return NotImplemented

    @property
    def ambient_dim(self):
        return self._dim

    @property
    def local_dim(self):
        return self._local_dim

    @property
    def parameterized(self):
        return self._parameterized

    def christoffel_matrices(self, base_point_coords):
        """
        Slow fallback implementation of computing christoffel matrices from normal vectors,
        which should be reimplemented explicitly for performance reasons
        (see for instance implementations on the  Sphere, Rotation Group, or Grassmannian)
        """
        dim = self._dim
        cm = np.empty([dim,dim,dim])
        basis = np.identity(dim)
        for i in range(dim):
            v_i = self.project_on_tangent_space(base_point_coords, basis[i])
            n_i = self.normal_vector(base_point_coords, v_i)
            cm[:,i,i] = - n_i
            for j in range(i,dim):
                v_j = self.project_on_tangent_space(base_point_coords, basis[j])
                n_ipj = self.normal_vector(base_point_coords, v_i + v_j)
                n_imj = self.normal_vector(base_point_coords, v_i - v_j)
                cm[:,i,j] = (n_imj - n_ipj)/4
                cm[:,j,i] = cm[:,i,j]
        return cm

    def christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
        """
        Slow fallback implementation of computing a linear combination of christoffel matrices,
        which should be reimplemented explicitly for performance reasons
        (see for instance implementations on the  Sphere, Rotation Group, or Grassmannian)
        """
        cm = self.christoffel_matrices(base_point_coords)
        return np.asmatrix(np.tensordot(coeffs, cm, axes=(0,0)))

class EuclideanSpace(Manifold):
    def __init__(self, d):
        self._dim=d
        self._local_dim=d
        self._description = "Euclidean Space R^{dim}".format(dim=self._dim)
        Manifold.__init__(self)

    def project_on_manifold(self, vector):
        return np.array(vector)

    def project_on_tangent_space(self, base_point_coords, vector):
        return np.array(vector)

    def geodesic_step(self, base_point_coords, tangent_vector_coords, step=1.0):
        new_base_point_coords = base_point_coords + step * tangent_vector_coords
        new_tangent_vector_coords  = np.array(tangent_vector_coords)
        return new_base_point_coords, new_tangent_vector_coords

    def normal_vector(self, base_point_coords, tangent_vector_coords):
        return np.zeros(tangent_vector_coords.shape)

    def christoffel_matrices(self, base_point_coords):
        return np.zeros((self._dim,)*3)

    def christoffel_matrices_lin_comb(self, base_point_coords):
        return np.asmatrix(np.zeros((self._dim,)*2))


class Sphere(Manifold):
    def __init__(self, d):
        self._dim=d+1
        self._local_dim = d
        self._description = "Sphere S^{d_s} in R^{d_r}".format(d_s=self._local_dim, d_r=self._dim)
        Manifold.__init__(self)

    def project_on_manifold(self, vector):
        norm = np.sqrt(np.sum(vector*vector,axis=1)).reshape([vector.shape[0],1])
        return vector/norm

    def project_on_tangent_space(self, base_point_coords, vector):
        pv = np.sum(base_point_coords*vector,axis=1).reshape([vector.shape[0],1])
        return vector - pv*base_point_coords

    def geodesic_step(self, base_point_coords, tangent_vector_coords, step=1.0):
        v_norm = np.sqrt(np.sum(tangent_vector_coords*tangent_vector_coords,axis=1)).reshape([tangent_vector_coords.shape[0],1])
        length = step*v_norm
        new_base_point_coords = np.cos(length)*base_point_coords + np.sin(length)*tangent_vector_coords/v_norm
        new_tangent_vector_coords = - v_norm*np.sin(length)*base_point_coords + np.cos(length)*tangent_vector_coords
        return new_base_point_coords, new_tangent_vector_coords

    def normal_vector(self, base_point_coords, tangent_vector_coords):
        norm2 = np.sum(tangent_vector_coords*tangent_vector_coords,axis=1)
        return -base_point_coords*norm2

    def christoffel_matrices(self, base_point_coords):
        dim = self._dim
        christoffel_matrices = np.empty((dim,)*3)
        for i in range(dim):
            christoffel_matrices[i,:,:] = np.eye(dim) * base_point_coords[i]
        return christoffel_matrices

    def christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
        return np.eye(self._dim)*(coeffs*base_point_coords).sum()

class Sphere3(Sphere):
    def __init__(self):
        Sphere.__init__(self,3)

    @staticmethod
    def compute_stereographicprojection(point_coords_4d):
        if point_coords_4d[0] > 0:
            point_coords_3d =  point_coords_4d[1:4]/(1+point_coords_4d[0])
        else:
            point_coords_3d = -point_coords_4d[1:4]/(1-point_coords_4d[0])
        return point_coords_3d

class Sphere2(Sphere):
    def __init__(self):
        Sphere.__init__(self,2)
        self._parameterized = True

    @staticmethod
    def compute_parameterization(spherical_coords):
        point_coords = np.empty([spherical_coords.shape[0],3])
        theta = spherical_coords[:,0]
        phi = spherical_coords[:,1]
        point_coords[:,0] = np.sin(theta) * np.cos(phi)
        point_coords[:,1] = np.sin(theta) * np.sin(phi)
        point_coords[:,2] = np.cos(theta)
        return point_coords

    @staticmethod
    def compute_inverse_parameterization(point_coords):
        local_coords = np.zeros([point_coords.shape[0],2])
        # theta
        local_coords[:,0] = np.arccos( point_coords[:,2] )
        # phi
        local_coords[:,1] = point_coords[:,1] / ( point_coords[:,0] - np.sqrt(point_coords[:,0]**2  + point_coords[:,1]**2) )
        local_coords[:,1] = 2*np.arctan(local_coords[:,1]) + np.pi
        mask = (point_coords[:,1]==0)*(0<=point_coords[:,0])
        local_coords[mask,1] = 0
        return local_coords

    @staticmethod
    def compute_jacobi_matrix(spherical_coords):
        theta = spherical_coords[:,0]
        phi = spherical_coords[:,1]
        jacobi_matrix = np.empty([spherical_coords.shape[0],3,2])
        # d_theat
        jacobi_matrix[:,0,0] =  np.cos(theta) * np.cos(phi)
        jacobi_matrix[:,1,0] =  np.cos(theta) * np.sin(phi)
        jacobi_matrix[:,2,0] = -np.sin(theta)
        # d_phi
        jacobi_matrix[:,0,1] = -np.sin(theta) * np.sin(phi)
        jacobi_matrix[:,1,1] =  np.sin(theta) * np.cos(phi)
        jacobi_matrix[:,2,1] = 0
        return jacobi_matrix

    @staticmethod
    def compute_inverse_riemannian_matrix(spherical_coords):
        inverse_riemannian_matrix = np.empty([spherical_coords.shape[0],2,2])
        theta = spherical_coords[:,0]
        # theta column
        inverse_riemannian_matrix[:,0,0] = 1.
        inverse_riemannian_matrix[:,1,0] = 0.
        # phi column
        inverse_riemannian_matrix[:,0,1] = 0.
        inverse_riemannian_matrix[:,1,1] = 1. / ( np.sin(theta) ** 2 )
        return inverse_riemannian_matrix

    @staticmethod
    def compute_christoffel_matrix_lin_comb_parameterization(spherical_coords,coeffs):
        theta = spherical_coords[:,0]
        christoffel_matrix = np.zeros([coeffs.shape[0],2,2])
        # theta_coeff * theta_ChristoffelMatrix
        christoffel_matrix[:,1,1] += -coeffs[:,0] * np.sin(theta) * np.cos(theta)
        # phi_coeff * phi_ChristoffelMatrix
        christoffel_matrix[:,0,1] += coeffs[:,1] / np.tan(theta)
        christoffel_matrix[:,1,0] += christoffel_matrix[:,0,1]
        return christoffel_matrix

class MatrixManifold(Manifold):
    @property
    def matrix_size(self):
        return self._matrix_size

    def coords_as_matrix(self, vector):
        return np.asmatrix(vector.reshape(self._matrix_size))

    def coords_as_vector(self, matrix):
        return np.asarray(matrix.reshape(self._dim))

class RotationGroup(MatrixManifold):
    def __init__(self, d):
        self._matrix_size = (d,d)
        self._dim=d**2
        self._local_dim=int(d*(d-1)/2)
        self._description = "Rotation Group SO({d_so}) in R^{d_r}".format(d_so=self._matrix_size[0], d_r=self._dim)
        Manifold.__init__(self)

    def project_on_manifold(self, vector):
        vector_m = vector.shape[0]
        projected_vector = np.empty([vector_m,self._dim])
        U, _, V = np.linalg.svd(vector.reshape([vector_m,self._matrix_size[0],self._matrix_size[1]]), full_matrices=True)
        m = np.matmul(U,V)
        det = np.linalg.det(m)
        for i in range(vector_m):#, v in enumerate(vector):
            #U, __, V = np.linalg.svd(self.coords_as_matrix(v), full_matrices=True)
            #m = np.matmul(U[i],V[i])
            if det[i]<0:#np.linalg.det(m) < 0:
                m[i,:,[0, 1]] = m[i,:,[1, 0]]
            projected_vector[i,:] = m[i,:,:].reshape(self._dim)#self.coords_as_vector(m)
        return projected_vector

    def project_on_tangent_space(self, base_point_coords, vector):
        projected_vector = np.empty([vector.shape[0],self._dim])
        m = base_point_coords.shape[0]
        m_p = base_point_coords.reshape(m,self._matrix_size[0],self._matrix_size[1])
        m_v = vector.reshape(m,self._matrix_size[0],self._matrix_size[1])
        return  ((m_v - np.matmul(m_p,np.matmul(m_v.transpose(0,2,1),m_p)))/2).reshape(m,self._dim)
        #for i in range(vector.shape[0]):
        #    m_p = (base_point_coords[i,:]).reshape(self._matrix_size)
        #    m_v = (vector[i,:]).reshape(self._matrix_size)
        #    projected_vector[i,:] = ((m_v - m_p @ m_v.transpose() @ m_p)/2).reshape(self._dim)
        #return projected_vector

    def geodesic_step(self, base_point_coords, tangent_vector_coords, step=1.0):
        m = base_point_coords.shape[0]
        #new_base_point_coords = np.empty([base_point_coords.shape[0],self._dim])
        #new_tangent_vector_coords = np.empty([tangent_vector_coords.shape[0],self._dim])
        m_p = base_point_coords.reshape(m,self._matrix_size[0],self._matrix_size[1])
        m_p_t = m_p.transpose(0,2,1)
        m_v = tangent_vector_coords.reshape(m,self._matrix_size[0],self._matrix_size[1])
        m_p_t_m_v = step*np.matmul(m_p_t,m_v)
        x = (np.sqrt(np.sum(np.sum(m_p_t_m_v**2,axis=1),axis=1)/2)).reshape(m,1,1)
        x[x==0]=1e-10
        expm1_step = np.sin(x)/x*m_p_t_m_v + (1-np.cos(x))/x**2*np.matmul(m_p_t_m_v,m_p_t_m_v)
        m_p_new = m_p + np.matmul(m_p,expm1_step)
        m_v_new = m_v + np.matmul(m_v,expm1_step)

        #for i in range(base_point_coords.shape[0]):
        #    norm_v = np.linalg.norm(tangent_vector_coords[i,:])
        #    if norm_v != 0 and step != 0:
        #        m_p = self.coords_as_matrix(base_point_coords[i,:])
        #        m_v = self.coords_as_matrix(tangent_vector_coords[i,:])
        #        expm_step = expm(step * m_p.transpose() * m_v)
        #        print(expm_step-expm1_step[i])
        #        new_base_point_coords[i,:] = self.coords_as_vector( m_p * expm_step )
        #        new_tangent_vector_coords[i,:] = self.coords_as_vector( m_v * expm_step )
        #    else:
        #        new_base_point_coords[i,:] = base_point_coords[i,:]
        #        new_tangent_vector_coords[i,:] = tangent_vector_coords[i,:]
        #return new_base_point_coords, new_tangent_vector_coords
        return m_p_new.reshape(m,self._dim), m_v_new.reshape(m,self._dim)

    def normal_vector(self, base_point_coords, tangent_vector_coords):
        m_pt = self.coords_as_matrix(base_point_coords).transpose()
        m_v = self.coords_as_matrix(tangent_vector_coords)
        return self.coords_as_vector( m_v * m_pt * m_v )

    def christoffel_matrices(self, base_point_coords):
        return cmanif.rotation_group_christoffel_matrices(self, base_point_coords)

    def christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
        return cmanif.rotation_group_christoffel_matrix_lin_comb(self, base_point_coords, coeffs)

class SO3(RotationGroup):
    def __init__(self):
        RotationGroup.__init__(self,3)
        self._parameterized = True

    @staticmethod
    def compute_parameterization(local_coords):
        coords = np.empty([local_coords.shape[0],9])
        for i, euler_coords in enumerate(local_coords):
            coords[i,:] = cmanif.SO3_parameterization(*euler_coords)
        return coords

    @staticmethod
    def compute_inverse_parameterization(point_coords):
        return cmanif.SO3_inverse_parameterization(point_coords.reshape([point_coords.shape[0],3,3]))
        #local_coords = np.empty([point_coords.shape[0],3])
        #for i, coords in enumerate(point_coords):
        #    local_coords[i,:] = cmanif.SO3_inverse_parameterization(coords.reshape([3,3]))
        #return local_coords

    @staticmethod
    def compute_jacobi_matrix(local_coords):
        return cmanif.SO3_jacobi_matrix(local_coords[:,0],local_coords[:,1],local_coords[:,2])
        #jacobi_matrix = np.empty([local_coords.shape[0],9,3])
        #for i, euler_coords in enumerate(local_coords):
        #    jacobi_matrix[i,:,:] = cmanif.SO3_jacobi_matrix(*euler_coords)
        #return jacobi_matrix

    @staticmethod
    def compute_inverse_riemannian_matrix(local_coords):
        return cmanif.SO3_inverse_riemannian_matrix(local_coords[:,1])
        #matrix = np.empty([local_coords.shape[0],3,3])
        #for i, euler_coords in enumerate(local_coords):
        #    matrix[i,:,:] = cmanif.SO3_inverse_riemannian_matrix(euler_coords[1])
        #return matrix

    @staticmethod
    def compute_christoffel_matrix_lin_comb_parameterization(local_coords,coeffs):
        matrix = np.empty([local_coords.shape[0],3,3])
        for i, euler_coords in enumerate(local_coords):
            matrix[i,:,:] = cmanif.SO3_christoffel_matrix_lin_comb_parameterization(euler_coords[1],*coeffs[i])
        return matrix

    @staticmethod
    def compute_quaternion_representation(point_coords):
        matrix = np.array(point_coords)
        quat = np.empty(4)
        trace = matrix[0] + matrix[4] + matrix[8]
        if trace > 0:
            s = 2*np.sqrt(trace+1)
            quat[0] = 0.25 * s
            quat[1] = (matrix[5] - matrix[7]) / s
            quat[2] = (matrix[6] - matrix[2]) / s
            quat[3] = (matrix[1] - matrix[3]) / s
        else:
            if (matrix[0] > matrix[4]) and (matrix[0] > matrix[8]):
                s = 2*np.sqrt(1.0 + matrix[0] - matrix[4] - matrix[8])
                quat[0] = (matrix[5] - matrix[7]) / s
                quat[1] = 0.25 * s
                quat[2] = (matrix[3] + matrix[1]) / s
                quat[3] = (matrix[6] + matrix[2]) / s
            else:
                if (matrix[4] > matrix[8]):
                    s = 2*np.sqrt(1.0 + matrix[4] - matrix[0] - matrix[8])
                    quat[0] = (matrix[6] - matrix[2]) / s
                    quat[1] = (matrix[3] + matrix[1]) / s
                    quat[2] = 0.25 * s
                    quat[3] = (matrix[7] + matrix[5]) / s
                else:
                    s = 2*np.sqrt(1.0 + matrix[8] - matrix[0] - matrix[4])
                    quat[0] = (matrix[1] - matrix[3]) / s
                    quat[1] = (matrix[6] + matrix[2]) / s
                    quat[2] = (matrix[7] + matrix[5]) / s
                    quat[3] = 0.25 * s
        return quat


class GrassmannianStiefelRepresentation(MatrixManifold):
    def __init__(self, k, d):
        self._matrix_size = (d,k)
        self._dim=k*d
        self._dim=int(k*(d-k))
        self._description = "Grassmannian G_{k_s},{d_s} in R^{d_r}".format(k_s=self._matrix_size[0], d_s=self._matrix_size[1], d_r=self._dim)
        Manifold.__init__(self)

    def project_on_manifold(self, vector):
        p = np.matrix(self.coords_as_matrix(vector))
        q, r = np.linalg.qr(p)
        return self.coords_as_vector(q)

    def project_on_tangent_space(self, base_point_coords, vector):
        m_p = self.coords_as_matrix(base_point_coords)
        m_v = self.coords_as_matrix(vector)
        return self.coords_as_vector(m_v - m_p*m_p.transpose()*m_v)

    def geodesic_step(self, base_point_coords, tangent_vector_coords, step=1.0):
        m_v = self.coords_as_matrix(tangent_vector_coords)
        u, s, v = np.linalg.svd(m_v,full_matrices=False)
        u = np.asmatrix(u)
        v = np.asmatrix(v.transpose())
        v_norm = np.linalg.norm(s)
        if v_norm != 0 and step != 0:
            m_pv = self.coords_as_matrix(base_point_coords)*v
            c_s = np.asmatrix(np.diag(np.cos(step*s)))
            s_s = np.asmatrix(np.diag(np.sin(step*s)))
            s = np.asmatrix(np.diag(s))
            new_base_point_coords =     self.coords_as_vector( ( m_pv  *c_s+u  *s_s)*v.transpose() )
            new_tangent_vector_coords = self.coords_as_vector( (-m_pv*s*s_s+u*s*c_s)*v.transpose() )
        else:
            new_base_point_coords = np.array(base_point_coords)
            new_tangent_vector_coords = np.array(tangent_vector_coords)
        return new_base_point_coords, new_tangent_vector_coords

    def normal_vector(self, base_point_coords, tangent_vector_coords):
        m_p = self.coords_as_matrix(base_point_coords)
        m_v = self.coords_as_matrix(tangent_vector_coords)
        return self.coords_as_vector(- m_p*m_v.transpose()*m_v)

    def christoffel_matrices(self, base_point_coortds):
        return cmanif.grassmannian_stiefel_rep_christoffel_matrices(self, base_point_coords)

    def christoffel_matrix_lin_comb(self, base_point_coords, coeffs):
        return cmanif.grassmannian_stiefel_rep_christoffel_matrix_lin_comb(self, base_point_coords, coeffs)


def generate_random_point_array(manifold, size):
    point_array = ManifoldPointArray(manifold)
    point_array.coords = np.random.randn(size, manifold.ambient_dim)
    return point_array


def generate_random_tangent_vector_array(point_array):
    tangent_vector_array = TangentVectorArray(point_array)
    tangent_vector_array.coords = np.random.randn(*point_array.coords.shape)
    return tangent_vector_array

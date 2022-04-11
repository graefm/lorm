from . import cnfsoft
import numpy as np

class SO3FourierCoefficients:
    def __init__(self,N):
        self._N = N
        self._fhat_array = np.zeros(self._dim(N),dtype=np.complex)

    @property
    def N(self):
        return self._N

    @property
    def array(self):
        return self._fhat_array

    @staticmethod
    def _dim(N):
        return int((2*N+1)*(2*N+2)*(2*N+3)/6)

    def __getitem__(self,index):
        if type(index[0]) == int:
            fhat_array_idx0 = self._fhat_array[self._dim(index[0]-1):self._dim(index[0])].reshape(2*index[0]+1,2*index[0]+1)
            if type(index[1]) == slice:
                s0 = -index[0] if index[1].start == None else index[1].start
                s1 =  index[0] if index[1].stop  == None else index[1].stop
            else:
                if type(index[1]) == int:
                    s0 = s1 = index[1]
            if type(index[2]) == slice:
                t0 = -index[0] if index[2].start == None else index[2].start
                t1 =  index[0] if index[2].stop  == None else index[2].stop
            else:
                if type(index[2]) == int:
                    t0 = t1 = index[2]
            return fhat_array_idx0[index[0]+s0:index[0]+s1+1,index[0]+t0:index[0]+t1+1]
        else:
            raise Exception('IndexError')

    def __setitem__(self,index,value):
        if type(index[0]) == int:
            fhat_array_idx0 = self._fhat_array[self._dim(index[0]-1):self._dim(index[0])].reshape(2*index[0]+1,2*index[0]+1)
            if type(index[1]) == slice:
                s0 = -index[0] if index[1].start == None else index[1].start
                s1 =  index[0] if index[1].stop  == None else index[1].stop
            else:
                if type(index[1]) == int:
                    s0 = s1 = index[1]
            if type(index[2]) == slice:
                t0 = -index[0] if index[2].start == None else index[2].start
                t1 =  index[0] if index[2].stop  == None else index[2].stop
            else:
                if type(index[2]) == int:
                    t0 = t1 = index[2]
            fhat_array_idx0[index[0]+s0:index[0]+s1+1,index[0]+t0:index[0]+t1+1] = value
        else:
            raise Exception('IndexError')

    def __iadd__(self,other):
        assert other._N <= self._N
        Nmin = other._N
        self._fhat_array[0:self._dim(Nmin)] += other._fhat_array
        return self

    def __isub__(self,other):
        assert other._N <= self._N
        Nmin = other._N
        self._fhat_array[0:self._dim(Nmin)] -= other._fhat_array
        return self

    def __mul__(self,other):
        Nmin = min(self._N,other._N)
        Nmax = max(self._N,other._N)
        mul = SO3FourierCoefficients(Nmax)
        mul._fhat_array[0:self._dim(Nmin)] = self._fhat_array[0:self._dim(Nmin)]*other._fhat_array[0:self._dim(Nmin)]
        return mul

class plan:
    def __init__(self,M,N,m=5):
        '''
        M number of points
        N bandwith
        '''
        self._M = M
        self._N = N
        self._N_internal = N+2
        self._m = m
        self._x = np.zeros([M,3])
        self._cplan = cnfsoft.plan(M,self._N_internal,m=self._m)

        self._coef_a = SO3FourierCoefficients(self._N_internal)
        self._coef_b = SO3FourierCoefficients(self._N_internal)
        self._coef_c = SO3FourierCoefficients(self._N_internal)

        self._coef_c1 = SO3FourierCoefficients(self._N_internal)
        self._coef_c2 = SO3FourierCoefficients(self._N_internal)

        for n in range(0,self._N_internal+1):
            for k1 in range(-n,n+1):
                for k2 in range(-n,n+1):
                    self._coef_a[n,k1,k2] =  n / ((n+1)*(2*n+1)) *  np.sqrt( (np.power(n+1,2) - np.power(k1,2)) * (np.power(n+1,2) - np.power(k2,2)) )
                    if n>0:
                        self._coef_b[n,k1,k2] = (1.*k1*k2)/(n*(n+1.))
                        self._coef_c[n,k1,k2] = (n+1.)/(n*(2*n+1.)) * np.sqrt( (n*n - k1*k1)*(n*n - k2*k2) )
                    self._coef_c1[n,k1,k2] = -1j*k1
                    self._coef_c2[n,k1,k2] = -1j*k2

    def set_local_coords(self,point_array_coords):
        '''
        (phi1,theta,phi2) in [-pi,pi) x [0,pi] x [-pi,pi)
        '''
        # cnfsoft uses coordinates (phi1,theta,phi2) in [-pi,pi) x [0,pi] x [-pi,pi)
        self._x = point_array_coords
        self._cplan.x = self._x
        self._cplan.precompute_x()

    def compute_Ymatrix_multiplication(self,fhat):
        '''
        returns  f = Y*fhat
        '''
        self._set_fhat(fhat)
        self._cplan.trafo()
        f = np.empty(self._M, dtype=np.complex)
        f[:] = self._cplan.f
        return f

    def compute_Ymatrix_adjoint_multiplication(self,f,N):
        '''
        returns  fhat = Y'*f
        '''
        self._cplan.f = f
        self._cplan.adjoint()
        return self._get_fhat(N)

    def compute_gradYmatrix_multiplication(self,fhat):
        '''
        returns gradf = gradY * fhat = (f_phi1, f_theta, f_phi2)
                f_phi1 =                 Y * Dphi1   * fhat
                f_theta = 1/sin(theta) * Y * Dtheta * fhat
                f_phi2 =                 Y * Dphi2   * fhat
        '''
        assert fhat.N <= self._N_internal - 1
        gradf = np.empty([self._M,3], dtype=np.complex)
        # f_phi1
        gradf[:,0] = self.compute_Ymatrix_multiplication(\
                    self._compute_Dphi1_matrix_multiplication(fhat) )
        # f_theta
        gradf[:,1] = self.compute_Ymatrix_multiplication(\
                    self._compute_Dtheta_matrix_multiplication(fhat) ) / np.sin(self._x[:,1])
        # f_phi2
        gradf[:,2] = self.compute_Ymatrix_multiplication(\
                    self._compute_Dphi2_matrix_multiplication(fhat) )
        return gradf

    def compute_gradYmatrix_adjoint_multiplication(self,f):
        '''
        return fhat = gradY' * (f_phi1, f_theta, f_phi2)   =  fhat_phi1 + fhat_theta + fhat_phi2
               fhat_phi1  = Dphi1'  * Y' * f_phi1
               fhat_theta = Dtheta' * Y' * 1 / sin(theta) * f_theta
               fhat_phi2  = Dphi2'  * Y' * f_phi2
        '''
        fhat_mult = self._compute_Dtheta_matrix_adjoint_multiplication(\
                     self.compute_Ymatrix_adjoint_multiplication( f[:,1] / np.sin(self._x[:,1]), self._N+1 ))
        fhat_mult += self._compute_Dphi1_matrix_adjoint_multiplication(\
                     self.compute_Ymatrix_adjoint_multiplication( f[:,0], self._N ))
        fhat_mult += self._compute_Dphi2_matrix_adjoint_multiplication(\
                     self.compute_Ymatrix_adjoint_multiplication( f[:,2], self._N ))
        return fhat_mult

    def compute_hessYmatrix_multiplication(self,fhat):
        '''
        returns hessf = hessY * fhat = (f_theta_theta, f_theta_phi, f_phi_phi)
                f_theta_theta = 1/sin(theta)^2 * (Y D_theta D_theta - cos(theta) Y D_theta)
        '''
        assert fhat.N <= self._N_internal - 2
        hessf = np.empty([self._M,6], dtype=np.complex)
        # f_phi1_phi1
        hessf[:,0]  = self.compute_Ymatrix_multiplication(\
                     self._compute_Dphi1_matrix_multiplication(\
                     self._compute_Dphi1_matrix_multiplication( fhat )))
        # f_phi1_theta
        hessf[:,1]  = self.compute_Ymatrix_multiplication(\
                     self._compute_Dtheta_matrix_multiplication(\
                     self._compute_Dphi1_matrix_multiplication( fhat ))) / np.sin(self._x[:,1])
        # f_phi1_phi2
        hessf[:,2] =  self.compute_Ymatrix_multiplication(\
                     self._compute_Dphi1_matrix_multiplication(\
                     self._compute_Dphi2_matrix_multiplication( fhat )))
        # f_theta_theta
        hessf[:,3]  = self.compute_Ymatrix_multiplication(\
                     self._compute_Dtheta_matrix_multiplication(\
                     self._compute_Dtheta_matrix_multiplication(fhat)))
        hessf[:,3] -= self.compute_Ymatrix_multiplication(\
                     self._compute_Dtheta_matrix_multiplication(fhat)) * np.cos(self._x[:,1])
        hessf[:,3] /= np.sin(self._x[:,1])**2
        # f_theta_phi2
        hessf[:,4]  = self.compute_Ymatrix_multiplication(\
                     self._compute_Dtheta_matrix_multiplication(\
                     self._compute_Dphi2_matrix_multiplication( fhat ))) / np.sin(self._x[:,1])
        # f_phi2_phi2
        hessf[:,5]  = self.compute_Ymatrix_multiplication(\
                     self._compute_Dphi2_matrix_multiplication(\
                     self._compute_Dphi2_matrix_multiplication( fhat )))
        return hessf

    def _get_fhat(self,N):
        fhat = SO3FourierCoefficients(N)
        cnfsoft._get_fhat(self,fhat)
        return fhat

    def _set_fhat(self,fhat):
        '''
        fhat is a SO3FourierCoefficients object
        it must have bandwith N lower or equal to the internal bandwith _N_internal
        '''
        cnfsoft._set_fhat(self,fhat)

    def _compute_Dtheta_matrix_multiplication(self,fhat):
        '''
        returns fhat_mult = D_theta * fhat
        '''
        assert fhat.N <= self._N_internal - 1
        fhat_mult = SO3FourierCoefficients(fhat.N+1)
        for n in range(0,fhat.N+1):
            #for k1 in range(-n,n+1):
            #    for k2 in range(-n,n+1):
            #        fhat_mult[n+1,k1,k2] += self._coef_a[n,k1,k2] * fhat[n,k1,k2]
            #        fhat_mult[n,k1,k2] -= self._coef_b[n,k1,k2] * fhat[n,k1,k2]
            #        if np.abs(k1) <= (n-1) and np.abs(k2) <= (n-1):
            #            fhat_mult[n-1,k1,k2] -= self._coef_c[n,k1,k2] * fhat[n,k1,k2]
            fhat_mult[n+1,-n:n,-n:n] += self._coef_a[n,-n:n,-n:n] * fhat[n,-n:n,-n:n]
            fhat_mult[n,  -n:n,-n:n] -= self._coef_b[n,-n:n,-n:n] * fhat[n,-n:n,-n:n]
            if n>0:
                fhat_mult[n-1,-n+1:n-1,-n+1:n-1] -= self._coef_c[n,-n+1:n-1,-n+1:n-1] * fhat[n,-n+1:n-1,-n+1:n-1]
        return fhat_mult

    def _compute_Dtheta_matrix_adjoint_multiplication(self,fhat):
        '''
        returns fhat_mult = D_theta' * fhat
        '''
        assert fhat.N <= self._N_internal + 1
        fhat_mult = SO3FourierCoefficients(fhat.N-1)
        for n in range(0,fhat.N):
            #for k1 in range(-n,n+1):
            #    for k2 in range(-n,n+1):
            #        fhat_mult(n,k1,k2) += coef_a(n,k1,k2) * fhat(n+1,k1,k2)
	        #        fhat_mult(n,k1,k2) -= coef_b(n,k1,k2) * fhat(n,k1,k2)
            #        if ( abs(k1) <= (n-1) && abs(k2) <= (n-1) ):
            #            fhat_mult(n,k1,k2) -= coef_c(n,k1,k2) * fhat(n-1,k1,k2)
            fhat_mult[n,-n  :n  ,-n  :n  ] += self._coef_a[n,-n  :n  ,-n  :n  ] * fhat[n+1,-n  :n  ,-n  :n  ]
            fhat_mult[n,-n  :n  ,-n  :n  ] -= self._coef_b[n,-n  :n  ,-n  :n  ] * fhat[n  ,-n  :n  ,-n  :n  ]
            if n>0:
                fhat_mult[n,-n+1:n-1,-n+1:n-1] -= self._coef_c[n,-n+1:n-1,-n+1:n-1] * fhat[n-1,-n+1:n-1,-n+1:n-1]
        return fhat_mult

    def _compute_Dphi1_matrix_multiplication(self,fhat):
        '''
        returns fhat_mult = D_phi1 * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SO3FourierCoefficients(fhat.N)
        for n in range(0,fhat.N+1):
            #for k in range(-n,n+1):
            #    fhat_mult[n,k] = -k1*1j *fhat[n,k]
            fhat_mult[n,:,:] = self._coef_c1[n,:,:] * fhat[n,:,:]
        return fhat_mult

    def _compute_Dphi1_matrix_adjoint_multiplication(self,fhat):
        '''
        returns fhat_mult = D_phi1' * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SO3FourierCoefficients(fhat.N)
        for n in range(0,fhat.N+1):
            #for k in range(-n,n+1):
            #    fhat_mult[n,k] = k1*1j *fhat[n,k]
            fhat_mult[n,:,:] = - self._coef_c1[n,:,:] * fhat[n,:,:]
        return fhat_mult

    def _compute_Dphi2_matrix_multiplication(self,fhat):
        '''
        returns fhat_mult = D_phi2 * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SO3FourierCoefficients(fhat.N)
        for n in range(0,fhat.N+1):
            #for k in range(-n,n+1):
            #    fhat_mult[n,k] = -k2*1j *fhat[n,k]
            fhat_mult[n,:,:] = self._coef_c2[n,:,:] * fhat[n,:,:]
        return fhat_mult

    def _compute_Dphi2_matrix_adjoint_multiplication(self,fhat):
        '''
        returns fhat_mult = D_phi2' * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SO3FourierCoefficients(fhat.N)
        for n in range(0,fhat.N+1):
            #for k in range(-n,n+1):
            #    fhat_mult[n,k] = k2*1j *fhat[n,k]
            fhat_mult[n,:,:] = - self._coef_c2[n,:,:] * fhat[n,:,:]
        return fhat_mult

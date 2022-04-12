from . import cnfsft
import numpy as np


class SphericalFourierCoefficients:
    def __init__(self, N):
        self._N = N
        self._fhat_array = np.zeros((N + 1) ** 2, dtype=np.complex)

    @property
    def N(self):
        return self._N

    @property
    def array(self):
        return self._fhat_array

    def __getitem__(self, index):
        if type(index[1]) == int:
            return self._fhat_array[(index[0]) ** 2 + (index[1] + index[0])]
        else:
            if type(index[1]) == slice:
                s0 = -index[0] if index[1].start == None else index[1].start
                s1 = index[0] if index[1].stop == None else index[1].stop
                a = (index[0]) ** 2 + index[0]
                return self._fhat_array[a + s0:a + s1 + 1]
            raise Exception('IndexError')

    def __setitem__(self, index, value):
        if type(index[1]) == int:
            self._fhat_array[(index[0]) ** 2 + (index[1] + index[0])] = value
        else:
            if type(index[1]) == slice:
                s0 = -index[0] if index[1].start == None else index[1].start
                s1 = index[0] if index[1].stop == None else index[1].stop
                a = (index[0]) ** 2 + index[0]
                self._fhat_array[a + s0:a + s1 + 1] = value
            else:
                raise Exception('IndexError')

    def __iadd__(self, other):
        assert other._N <= self._N
        Nmin = other._N
        self._fhat_array[0:(Nmin + 1) ** 2] += other._fhat_array
        return self

    def __isub__(self, other):
        assert other._N <= self._N
        Nmin = other._N
        self._fhat_array[0:(Nmin + 1) ** 2] -= other._fhat_array
        return self

    def __mul__(self, other):
        Nmin = min(self._N, other._N)
        Nmax = max(self._N, other._N)
        mul = SphericalFourierCoefficients(Nmax)
        mul._fhat_array[0:(Nmin + 1) ** 2] = self._fhat_array[0:(Nmin + 1) ** 2] * other._fhat_array[0:(Nmin + 1) ** 2]
        return mul


class plan:
    def __init__(self, M, N):
        '''
        M number of points
        N bandwith
        '''
        self._M = M
        self._N = N
        self._N_internal = N + 2
        self._x = np.zeros([M, 2])
        self._cplan = cnfsft.plan(M, self._N_internal)
        self.direct = False

        self._coef_a = SphericalFourierCoefficients(self._N_internal)
        self._coef_b = SphericalFourierCoefficients(self._N_internal)
        self._coef_c = SphericalFourierCoefficients(self._N_internal)

        for n in range(0, self._N_internal + 1):
            for k in range(-n, n + 1):
                self._coef_a[n, k] = n * np.sqrt(
                    (np.power(n + 1, 2) - np.power(k, 2)) / ((2. * n + 1.) * (2. * n + 3.)))
                self._coef_b[n, k] = (n + 1) * np.sqrt(
                    (np.power(n, 2) - np.power(k, 2)) / ((2. * n - 1.) * (2. * n + 1.)))
                self._coef_c[n, k] = k * 1j

    def set_local_coords(self, point_array_coords):
        '''
        (theta,phi) in [0,pi]x[0,2 pi)
        '''
        # cnfsft uses coordinates (phi,theta) in [-1/2,1/2) x [0,1/2]
        self._x[:] = point_array_coords
        self._cplan.x[:, 0] = self._x[:, 1] / (2 * np.pi)
        self._cplan.x[:, 1] = self._x[:, 0] / (2 * np.pi)
        self._cplan.precompute_x()

    def compute_Ymatrix_multiplication(self, fhat):
        '''
        returns  f = Y*fhat
        '''
        self._set_fhat(fhat)
        self._cplan.trafo(direct=self.direct)
        f = np.empty(self._M, dtype=np.complex)
        f[:] = self._cplan.f
        return f

    def compute_Ymatrix_adjoint_multiplication(self, f, N):
        '''
        returns  fhat = Y'*f
        '''
        self._cplan.f = f
        self._cplan.adjoint(direct=self.direct)
        return self._get_fhat(N)

    def compute_gradYmatrix_multiplication(self, fhat):
        '''
        returns gradf = gradY * fhat = (f_theta, f_phi)
                f_thata = 1/sin(theta) * Y * Dtheta * fhat
                f_phi =                  Y * Dphi   * fhat
        '''
        assert fhat.N <= self._N_internal - 1
        gradf = np.empty([self._M, 2], dtype=np.complex)
        # f_theta
        gradf[:, 0] = self.compute_Ymatrix_multiplication(
            self._compute_Dtheta_matrix_multiplication(fhat)) / np.sin(self._x[:, 0])
        # f_phi
        gradf[:, 1] = self.compute_Ymatrix_multiplication(
            self._compute_Dphi_matrix_multiplication(fhat))
        return gradf

    def compute_gradYmatrix_adjoint_multiplication(self, f):
        '''
        return fhat = gradY' * (f_theta, f_phi)   =  fhat_theta + fhat_phi
               fhat_theta = Dtheta' * Y' * 1 / sin(theta) * f_theta
               fhat_phi   = Dphi'   * Y' * f_phi
        '''
        fhat_mult = self._compute_Dtheta_matrix_adjoint_multiplication(
            self.compute_Ymatrix_adjoint_multiplication(f[:, 0] / np.sin(self._x[:, 0]), self._N + 1))
        fhat_mult += self._compute_Dphi_matrix_adjoint_multiplication(
            self.compute_Ymatrix_adjoint_multiplication(f[:, 1], self._N))
        return fhat_mult

    def compute_hessYmatrix_multiplication(self, fhat):
        '''
        returns hessf = hessY * fhat = (f_theta_theta, f_theta_phi, f_phi_phi)
                f_theta_theta = 1/sin(theta)^2 * (Y D_theta D_theta - cos(theta) Y D_theta)
        '''
        assert fhat.N <= self._N_internal - 2
        hessf = np.empty([self._M, 3], dtype=np.complex)
        # f_theta_theta
        hessf[:, 0] = self.compute_Ymatrix_multiplication(
            self._compute_Dtheta_matrix_multiplication(
                self._compute_Dtheta_matrix_multiplication(fhat)))
        hessf[:, 0] -= self.compute_Ymatrix_multiplication(
            self._compute_Dtheta_matrix_multiplication(fhat)) * np.cos(self._x[:, 0])
        hessf[:, 0] /= np.sin(self._x[:, 0]) ** 2
        # f_theta_phi
        hessf[:, 1] = self.compute_Ymatrix_multiplication(
            self._compute_Dtheta_matrix_multiplication(
                self._compute_Dphi_matrix_multiplication(fhat))) / np.sin(self._x[:, 0])
        # f_phi_phi
        hessf[:, 2] = self.compute_Ymatrix_multiplication(
            self._compute_Dphi_matrix_multiplication(
                self._compute_Dphi_matrix_multiplication(fhat)))
        return hessf

    def _get_fhat(self, N):
        assert N <= self._N_internal
        # 0 <= k <= N, -k <= l <= k
        # cplan f_hat k^l = [N-l+1,N+k+1]
        fhat = SphericalFourierCoefficients(N)
        for n in range(0, N + 1):
            #            for k in range(-n,n+1):
            #                fhat[n,k] = self._cplan.f_hat[self._N_internal-k+1,self._N_internal+n+1]
            fhat[n, -n:n] = np.flip(
                self._cplan.f_hat[self._N_internal - n + 1:self._N_internal + n + 2, self._N_internal + n + 1])
        return fhat

    def _set_fhat(self, fhat):
        '''
        fhat is a SphericalFourierCoefficients object
        it must have bandwith N lower or equal to the internal bandwith _N_internal
        '''
        assert fhat.N <= self._N_internal
        self._cplan.f_hat[:] = 0
        # 0 <= k <= N, -k <= l <= k
        # cplan f_hat k^l = [N-l+1,N+k+1]
        for n in range(0, fhat.N + 1):
            #            for k in range(-n,n+1):
            #                self._cplan.f_hat[self._N_internal-k+1,self._N_internal+n+1]=fhat[n,k]
            self._cplan.f_hat[self._N_internal - n + 1:self._N_internal + n + 2, self._N_internal + n + 1] = np.flip(
                fhat[n, -n:n])

    def _compute_Dtheta_matrix_multiplication(self, fhat):
        '''
        returns fhat_mult = D_theta * fhat
        '''
        assert fhat.N <= self._N_internal - 1
        fhat_mult = SphericalFourierCoefficients(fhat.N + 1)
        for n in range(0, fhat.N + 1):
            # for k in range(-n,n+1):
            #    fhat_mult[n+1,k] = self.coef_a(n,k) * fhat[n,k]
            #    if np.abs(k) <= (n-1):
            #        fhat_mult[n-1,k] -= self.coef_b(n,k) * fhat[n,k]
            fhat_mult[n + 1, -n:n] = self._coef_a[n, -n:n] * fhat[n, -n:n]
            fhat_mult[n - 1, -n + 1:n - 1] -= self._coef_b[n, -n + 1:n - 1] * fhat[n, -n + 1:n - 1]
        return fhat_mult

    def _compute_Dtheta_matrix_adjoint_multiplication(self, fhat):
        '''
        returns fhat_mult = D_theta' * fhat
        '''
        assert fhat.N <= self._N_internal + 1
        fhat_mult = SphericalFourierCoefficients(fhat.N - 1)
        for n in range(0, fhat.N):
            # for k in range(-n,n+1):
            #    fhat_mult[n,k] =  coef_a[n,k] * fhat[n+1,k]
            #    if ( np.abs(k) <= (n-1)):
            #        fhat_mult[n,k] -=  coef_b[n,k] * fhat[n-1,k]
            fhat_mult[n, -n:n] = self._coef_a[n, -n:n] * fhat[n + 1, -n:n]
            fhat_mult[n, -n + 1:n - 1] -= self._coef_b[n, -n + 1:n - 1] * fhat[n - 1, -n + 1:n - 1]
        return fhat_mult

    def _compute_Dphi_matrix_multiplication(self, fhat):
        '''
        returns fhat_mult = D_phi * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SphericalFourierCoefficients(fhat.N)
        for n in range(0, fhat.N + 1):
            # for k in range(-n,n+1):
            #    fhat_mult[n,k] = k*1j *fhat[n,k]
            fhat_mult[n, :] = self._coef_c[n, :] * fhat[n, :]
        return fhat_mult

    def _compute_Dphi_matrix_adjoint_multiplication(self, fhat):
        '''
        returns fhat_mult = D_phi' * fhat
        '''
        assert fhat.N <= self._N_internal
        fhat_mult = SphericalFourierCoefficients(fhat.N)
        for n in range(0, fhat.N + 1):
            # for k in range(-n,n+1):
            #    fhat_mult[n,k] = -k*1j *fhat[n,k]
            fhat_mult[n, :] = - self._coef_c[n, :] * fhat[n, :]
        return fhat_mult

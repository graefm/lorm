from . import cnfft
import numpy as np
import scipy as sp
from scipy import special


def leg_matrix(k,N,x):
    '''' L = (P_m^k(x_i))_{i=1,...,M; m=|k|,...,N} '''
    K = np.abs(k)
    L = np.zeros((len(x),N-K+1))
    # initial values
    if k>=0:
        L[:,0] = (-1)**K*sp.special.factorial2(2*K-1)*np.power(1-x**2,K/2)
        if N>K:
            L[:,1] = (2*K+1)*x*L[:,0]
    else:
        L[:,0] = sp.special.factorial2(2*K-1)/sp.special.factorial(2*K)*np.power(1-x**2,K/2)
        if N>K:
            L[:,1] = x*L[:,0]
    # three term recurrence relations
    # (m-k+1)P_{m+1}^{k}(x) = (2m+1)xP_{m}^{k}(x) - (m+k)P_{m-1}^k(x)
    if N > K+1:
        for m in range(K+1,N):
            L[:,m-K+1] = ((2*m+1)*x*L[:,m-K]-(m+k)*L[:,m-K-1])/(m-k+1)
            # L[:,m-K] = sp.special.lpmv(k,m,x_eval)
    return L

def leg_hat_to_exp_hat_transition_matrix(k,N):
    ''' P_m^k(\cos(\theta)) = \sum_{n=-N-1}^N \hat p_m^k(n) exp(-i n \theta) for \theta \in [0,\pi]
        L2E = (p_m^k(n))_{n=-N-1,...,N; m=|k|,...,N} '''
    N_exp=2*N+2
    K = np.abs(k)
    L2E = np.zeros((N_exp,N-K+1),dtype=np.complex)
    sgn = np.sign(np.linspace(-np.pi,np.pi*(1-2/N_exp),N_exp)+1e-10) #
    L = leg_matrix(k,N,np.cos(np.linspace(0,N_exp-1,N_exp)/N_exp*(2*np.pi)))
    for m in range(K,N+1):
        # for odd orders k we need to compensate for the factor \sqrt(\sin(\theta)^2)
        # in order to get a trigonometric polynomial on [-\pi,\pi)
        L2E[:,m-K] = np.fft.fftshift(np.fft.ifft((-sgn)**k*L[:,m-K]))
    return L2E


class plan:
    def __init__(self,M,N,m=5):
        '''
        M number of points
        N bandwith
        '''
        self._M = M
        self._m = m
        self._N = N
        self._N_internal = N
        self._N_exp = (2*self._N_internal+2,)*4
        n_exp = [2*i for i in self._N_exp]
        self.nfft_c_plan = cnfft.plan(M,self._N_exp,n_exp,self._m)

        # precompute legendre to exponential transition coefficients
        self._leg_to_exp = [leg_hat_to_exp_hat_transition_matrix(k,self._N_internal) for k in range(-self._N_internal,self._N_internal+1)]
        # precompute spherical harmonics factors
        self._factors = [np.array([np.sqrt((2*m+1)/(4*np.pi)*sp.math.factorial(m-k)/sp.math.factorial(m+k)) for m in range(np.abs(k),N+1)]) for k in range(-self._N_internal,self._N_internal+1)]

    @staticmethod
    def linearized_index(N,K0,M0,K1,M1):
        i=0
        if M0 < np.abs(K0) or M1 < np.abs(K1) or M0 > N or M1 > N:
            raise Exception('IndexError')
        for k0 in range(-N,N+1):
            for k1 in range(-N,N+1):
                if k0<K0 or k1<K1:
                    i += (N-np.abs(k0)+1)*(N-np.abs(k1)+1)
                else:
                    i+= (N-np.abs(k1)+1)*(M0-np.abs(k0)) + (M1-np.abs(k1))
                    return i

    def set_local_coords(self,point_array_coords):
        '''
        (theta1,phi1,theta2,phi2) in [0,pi]x[0,2 pi)x[0,pi]x[0,2 pi)
        '''
        self.nfft_c_plan.x[:] = point_array_coords/2/np.pi
        self.nfft_c_plan.precompute_x()

    def compute_Ymatrix(self):
        ''' Y = ( Y_k^m(\theta^1_i,\phi^1_i) Y_l^m(\theta_2_i,\phi^2_i) )_{i=1,...,M; k,l=-N,...,N, m=|k|,...,N, n=|l|,...,N}
            x: sampling points (\theta_1,\phi_1,theta_2,\phi_2) in [0,\pi]x[0,2\pi)x[0,\pi]x[0,2\pi) '''
        x = 2*np.pi*self.nfft_c_plan.x
        Y = np.zeros((len(x),(N+1)**4),dtype=np.complex)
        i = 0
        for k_phi_1 in range(-N,N+1):
            L_k_phi_1 = leg_matrix(k_phi_1,N,np.cos(x[:,0]))
            e_k_phi_1 = np.cos(k_phi_1*x[:,1]) + 1j*np.sin(k_phi_1*x[:,1])
            s1 = N-np.abs(k_phi_1)+1
            Y_k_phi_1 = e_k_phi_1.reshape(len(x),1)*L_k_phi_1*self._factors[k_phi_1+N][:].reshape(1,s1)
            for k_phi_2 in range(-N,N+1):
                L_k_phi_2 = leg_matrix(k_phi_2,N,np.cos(x[:,2]))
                e_k_phi_2 = np.cos(k_phi_2*x[:,3]) + 1j*np.sin(k_phi_2*x[:,3])
                s2 = N-np.abs(k_phi_2)+1
                Y_k_phi_2 = e_k_phi_2.reshape(len(x),1)*L_k_phi_2*self._factors[k_phi_2+N][:].reshape(1,s2)
                temp = np.zeros((len(x),s1,s2),dtype=np.complex)
                temp[:,:,:]  = Y_k_phi_1.reshape(len(x),s1,1)
                temp[:,:,:] *= Y_k_phi_2.reshape(len(x),1,s2)
                #for m0 in range(np.abs(k_phi_1),N+1):
                #    Y_k_phi_1m0 = self._factors[k_phi_1+N][m0-np.abs(k_phi_1)] * e_k_phi_1*L_k_phi_1[:,m0-np.abs(k_phi_1)].reshape(len(x),1)
                #    for m1 in range(np.abs(k_phi_2),N+1):
                #        Y[:,i] = self._factors[k_phi_2+N][m1-np.abs(k_phi_2)] * e_k_phi_2*L_k_phi_2[:,m1-np.abs(k_phi_2)]
                #        Y[:,i] *= Y_k_phi_1m0
                #        i+=1
                Y[:,i:i+s1*s2] = temp.reshape(len(x),s1*s2)
                i += s1*s2
        return Y

    def compute_Ymatrix_multiplication_direct(self,f_hat):
        '''
        return f = Y*f_hat
        '''
        x = 2*np.pi*self.nfft_c_plan.x
        f = np.zeros(len(x),dtype=np.complex)
        i = 0
        N = self._N_internal
        for k_phi_1 in range(-N,N+1):
            L_k_phi_1 = leg_matrix(k_phi_1,N,np.cos(x[:,0]))
            e_k_phi_1 = np.cos(k_phi_1*x[:,1]) + 1j*np.sin(k_phi_1*x[:,1])
            s1 = N-np.abs(k_phi_1)+1
            Y_k_phi_1 = e_k_phi_1.reshape(len(x),1)*L_k_phi_1*self._factors[k_phi_1+N][:].reshape(1,s1)
            for k_phi_2 in range(-N,N+1):
                L_k_phi_2 = leg_matrix(k_phi_2,N,np.cos(x[:,2]))
                e_k_phi_2 = np.cos(k_phi_2*x[:,3]) + 1j*np.sin(k_phi_2*x[:,3])
                s2 = N-np.abs(k_phi_2)+1
                Y_k_phi_2 = e_k_phi_2.reshape(len(x),1)*L_k_phi_2*self._factors[k_phi_2+N][:].reshape(1,s2)
                temp = np.zeros((len(x),s1,s2),dtype=np.complex)
                temp[:,:,:]  = Y_k_phi_1.reshape(len(x),s1,1)
                temp[:,:,:] *= Y_k_phi_2.reshape(len(x),1,s2)
                #for m0 in range(np.abs(k_phi_1),N+1):
                #    Y_k_phi_1m0 = self._factors[k_phi_1+N][m0-np.abs(k_phi_1)] * e_k_phi_1*L_k_phi_1[:,m0-np.abs(k_phi_1)].reshape(len(x),1)
                #    for m1 in range(np.abs(k_phi_2),N+1):
                #        Y[:,i] = self._factors[k_phi_2+N][m1-np.abs(k_phi_2)] * e_k_phi_2*L_k_phi_2[:,m1-np.abs(k_phi_2)]
                #        Y[:,i] *= Y_k_phi_1m0
                #        i+=1
                #Y[:,i:i+s1*s2] = temp.reshape(len(x),s1*s2)
                f += np.dot(temp.reshape(len(x),s1*s2),f_hat[i:i+s1*s2])
                i += s1*s2
        return f

    def compute_Ymatrix_adjoint_multiplication_direct(self,f):
        '''
        returns f_hat = Y'*f
        '''
        x = 2*np.pi*self.nfft_c_plan.x
        N = self._N_internal
        f_hat = np.zeros((N+1)**4,dtype=np.complex)
        i = 0
        for k_phi_1 in range(-N,N+1):
            L_k_phi_1 = leg_matrix(k_phi_1,N,np.cos(x[:,0]))
            e_k_phi_1 = np.cos(k_phi_1*x[:,1]) + 1j*np.sin(k_phi_1*x[:,1])
            s1 = N-np.abs(k_phi_1)+1
            Y_k_phi_1 = e_k_phi_1.reshape(len(x),1)*L_k_phi_1*self._factors[k_phi_1+N][:].reshape(1,s1)
            for k_phi_2 in range(-N,N+1):
                L_k_phi_2 = leg_matrix(k_phi_2,N,np.cos(x[:,2]))
                e_k_phi_2 = np.cos(k_phi_2*x[:,3]) + 1j*np.sin(k_phi_2*x[:,3])
                s2 = N-np.abs(k_phi_2)+1
                Y_k_phi_2 = e_k_phi_2.reshape(len(x),1)*L_k_phi_2*self._factors[k_phi_2+N][:].reshape(1,s2)
                temp = np.zeros((len(x),s1,s2),dtype=np.complex)
                temp[:,:,:]  = Y_k_phi_1.reshape(len(x),s1,1)
                temp[:,:,:] *= Y_k_phi_2.reshape(len(x),1,s2)
                #for m0 in range(np.abs(k_phi_1),N+1):
                #    Y_k_phi_1m0 = self._factors[k_phi_1+N][m0-np.abs(k_phi_1)] * e_k_phi_1*L_k_phi_1[:,m0-np.abs(k_phi_1)].reshape(len(x),1)
                #    for m1 in range(np.abs(k_phi_2),N+1):
                #        Y[:,i] = self._factors[k_phi_2+N][m1-np.abs(k_phi_2)] * e_k_phi_2*L_k_phi_2[:,m1-np.abs(k_phi_2)]
                #        Y[:,i] *= Y_k_phi_1m0
                #        i+=1
                #Y[:,i:i+s1*s2] = temp.reshape(len(x),s1*s2)
                f_hat[i:i+s1*s2] = np.dot(f,temp.conj().reshape(len(x),s1*s2))
                i += s1*s2
        return f_hat

    def compute_Ymatrix_multiplication(self,f_hat):
        '''
        returns  f = Y*fhat
        '''
        # compute exponential Fourier coefficients
        exp_hat = np.zeros(self._N_exp,dtype=np.complex)
        N = self._N_internal
        i = 0
        for k_phi_1 in range(-N,N+1):
            s1 = N-np.abs(k_phi_1)+1
            pf_k_phi_1 = self._leg_to_exp[N+k_phi_1]*self._factors[N+k_phi_1].reshape(1,s1) # leg_hat_to_exp_hat_transition_matrix(k_phi_1,N)
            for k_phi_2 in range(-N,N+1):
                s2 = N-np.abs(k_phi_2)+1
                pf_k_phi_2 = self._leg_to_exp[N+k_phi_2]*self._factors[N+k_phi_2].reshape(1,s2)#leg_hat_to_exp_hat_transition_matrix(k_phi_2,N)
                #g_m__k_theta_2 = np.zeros((s1,2*N+2),dtype=np.complex)
                #i = index(N,k_phi_1,np.abs(k_phi_1),k_phi_2,np.abs(k_phi_2))
                #for m in range(np.abs(k_phi_1),N+1):
                #    for k_theta_2 in range(-N,N+1):
                #        for n in range(np.abs(k_phi_2),N+1):
                #            g_m__k_theta_2[m-np.abs(k_phi_1),k_theta_2+N+1] += p_k_phi_2[k_theta_2+N+1,n-np.abs(k_phi_2)] * f_hat[index(N,k_phi_1,m, k_phi_2,n)]
                #g_m__k_theta_2 = np.dot(f_hat[i:i+s1*s2].reshape(s1,s2),p_k_phi_2.transpose())
                #g_k_theta_1__k_theta_2 = np.zeros((2*N+2,2*N+2),dtype=np.complex)
                #for k_theta_1 in range(-N,N+1):
                #    for k_theta_2 in range(-N,N+1):
                #        for m in range(np.abs(k_phi_1),N+1):
                #            g_k_theta_1__k_theta_2[k_theta_1+N+1,k_theta_2+N+1] += p_k_phi_1[k_theta_1+N+1,m-np.abs(k_phi_1)]*g_m__k_theta_2[m-np.abs(k_phi_1),k_theta_2+N+1]
                #exp_hat[:,-k_phi_1+N+1,:,-k_phi_2+N+1] = np.dot(p_k_phi_1,g_m__k_theta_2)
                exp_hat[:,-k_phi_1+N+1,:,-k_phi_2+N+1] = np.dot(pf_k_phi_1,np.dot(f_hat[i:i+s1*s2].reshape(s1,s2),pf_k_phi_2.transpose()))
                i += s1*s2
        # compute nfft
        self.nfft_c_plan.f_hat = exp_hat
        self.nfft_c_plan.trafo()
        # get function values and return
        f = np.empty(self._M,dtype=np.complex)
        f[:] = self.nfft_c_plan.f
        return f


    def compute_Ymatrix_adjoint_multiplication(self,f):
        '''
        returns  f_hat = Y'*f
        '''
        # compute adjoint nfft
        self.nfft_c_plan.f = f
        self.nfft_c_plan.adjoint()
        exp_hat = self.nfft_c_plan.f_hat
        # compute Fourier coefficients
        N = self._N_internal
        f_hat = np.empty((N+1)**4,dtype=np.complex)
        i = 0
        for k_phi_1 in range(-N,N+1):
            s1 = N-np.abs(k_phi_1)+1
            pf_k_phi_1 = self._leg_to_exp[N+k_phi_1]*self._factors[N+k_phi_1].reshape(1,s1)#leg_hat_to_exp_hat_transition_matrix(k_phi_1,N)
            for k_phi_2 in range(-N,N+1):
                s2 = N-np.abs(k_phi_2)+1
                pf_k_phi_2 = self._leg_to_exp[N+k_phi_2]*self._factors[N+k_phi_2].reshape(1,s2)#leg_hat_to_exp_hat_transition_matrix(k_phi_2,N)
                #f_m1__k_theta_1 = np.zeros((N-np.abs(k_phi_2)+1,2*N+2),dtype=np.complex)
                #for m1 in range(np.abs(k_phi_2),N+1):
                #    for k_theta_1 in range(-N,N+1):
                #        for l in range(-N,N+1):
                #            f_m1__k_theta_1[m1-np.abs(k_phi_2),k_theta_1+N+1] += p_k_phi_2[l+N+1,m1-np.abs(k_phi_2)] * exp_hat[k_theta_1+N+1,-k_phi_1+N+1,-l+N+1,-k_phi_2+N+1]
                #f_m1__k_theta_1 = np.dot(exp_hat[:,-k_phi_1+N+1,:,-k_phi_2+N+1],p_k_phi_2.conj()).transpose()
                #f_m0__m1 = np.zeros((N-np.abs(k_phi_1)+1,N-np.abs(k_phi_2)+1),dtype=np.complex)
                #for m0 in range(np.abs(k_phi_1),N+1):
                #    for m1 in range(np.abs(k_phi_2),N+1):
                #        for k in range(-N,N+1):
                #            f_m0__m1[m0-np.abs(k_phi_1),m1-np.abs(k_phi_2)] += p_k_phi_1[k+N+1,m0-np.abs(k_phi_1)].conj() * f_m1__k_theta_1[m1-np.abs(k_phi_2),k+N+1]
                #f_m0__m1 = np.dot(f_m1__k_theta_1,p_k_phi_1.conj()).transpose()
                #for m0 in range(np.abs(k_phi_1),N+1):
                #    for m1 in range(np.abs(k_phi_2),N+1):
                #        f_hat[index(N, k_phi_1,m0, k_phi_2,m1)] = f_m0__m1[m0-np.abs(k_phi_1),m1-np.abs(k_phi_2)]
                #f_hat[i:i+s1*s2] = np.dot(f_m1__k_theta_1,p_k_phi_1.conj()).transpose().ravel()
                f_hat[i:i+s1*s2] = np.dot(pf_k_phi_1.conj().transpose(),np.dot(exp_hat[:,-k_phi_1+N+1,:,-k_phi_2+N+1],pf_k_phi_2.conj())).ravel()
                i += s1*s2
        return f_hat

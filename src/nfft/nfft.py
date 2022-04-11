from . import cnfft

class NFFT2D(cnfft.plan):
    def __init__(self,M,Nx,Ny,m=5):
        '''
        M number of points
        Nx bandwith in x
        Ny bandwith in y
        '''
        self._N = [Nx,Ny]
        self._n = [2*Nx, 2*Ny]
        self._m = m
        cnfft.plan.__init__(self,M,self._N,self._n,self._m)

    pass

class NFFT3D(cnfft.plan):
    def __init__(self,M,Nx,Ny,Nz,m=5):
        '''
        M number of points
        Nx bandwith in x
        Ny bandwith in y
        Nz bandwith in z
        '''
        self._N = [Nx,Ny,Nz]
        self._n = [2*Nx, 2*Ny, 2*Nz]
        self._m = m
        cnfft.plan.__init__(self,M,self._N,self._n,self._m)

    pass

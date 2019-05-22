from . import cnfft

class NFFT2D(cnfft.plan):
    def __init__(self,M,Nx,Ny):
        '''
        M number of points
        Nx bandwith in x
        Ny bandwith in y
        '''
        self._N = [Nx,Ny]
        self._n = [2*Nx, 2*Ny]
        self._m = 7
        cnfft.plan.__init__(self,M,self._N,self._n,self._m)

    pass

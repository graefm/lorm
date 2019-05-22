import lorm
from lorm import discrepancySO3
import numpy as np
import pylab as pl
%matplotlib inline
import matplotlib.pyplot as plt

so3 = lorm.manif.SO3()

p = lorm.manif.ManifoldPointArrayParameterized(so3)
M = 200
p.coords = np.random.randn(M,9)

N = 8
energy = discrepancySO3.plan(M,N)
for n in range(N+1):
    energy._lambda_hat[n,:,:] = 1./(2.*n+1)**5
    # integrate(legendre_p_n(x),x,0,1)
int_pn = [1,1/2,0,-1/8,0,1/16,0,-5/128,0,7/256,0,-21/1024,0,33/2048,0,-429/32768,0,715/65536,0,-2431/262144,0]
# integrate(legendre_p_n(x),x,-1,0)
#int_pn = [1,-1/2,0,1/8,0,-1/16,0,5/128,0,-7/256,0,21/1024,0,-33/2048,0,429/32768,0,-715/65536,0,2431/262144,0]
# integrate(x * legendre_p_n(x),0,1)
#int_pn = [1/2,1/3,1/8,0,-1/48,0,1/128,0,-1/256,0,7/3072,0,-3/2048,0,33/32768,0,-143/196608,0,143/262144,0,-221/524288]
# integrate( (pi/2-arccos(x))^2 * legendre_p_n(x),0,1)
#int_pn = [0.4674011002723398,0.3668502750680843,0.2222222222222223,0.1010531421917553,0.0355555555555555,0.01484661888127219,0.01160997732426307,0.008973288625497218,0.005159989921895382,0.003068786426492212,0.002729250867771737]
# integrate( (pi-arccos(x)) * legendre_p_n(x),-1,1)
#int_pn =[3.141592653589793,0.7853981633974448,0,0.04908738521234036,0,0.01227184630308148,0,0.004793689962076684,0,0.002348908082027755,0,0.001321260789556922,0,0.815675979112751*10**-3,0,0.538468280884195*10**-3,0,0.373970709757952*10**-3,0,0.267526379439437*10**-3,0]
for n in range(N+1):
    energy._mu_hat[n,0,0] = int_pn[n]
energy._weights = np.real(self.energy._mu_hat[0,0,0])* np.ones([int(M),1],dtype=float) / int(M)
cgMethod = lorm.optim.ConjugateGradientMethod(max_iter=100)
p_new = p
p_new = cgMethod.run(energy,p_new)
dist = np.zeros([M,M])
for i,x in enumerate(p_new.coords):
    for j,y in enumerate(p_new.coords):
        calpha = (np.trace(np.asmatrix(x.reshape(3,3)).transpose()*np.asmatrix(y.reshape(3,3)))-1)/2
        if (calpha>1):
            dist[i,j] = 0
        else:
            if (calpha<-1):
                dist[i,j] = np.pi
            else:
                dist[i,j] = np.arccos(calpha)
#print(dist)

plt.hist(dist.ravel(), bins=200)

from nfft import nfft
import numpy as np
import matplotlib.pyplot as plt
%matplotlib

Mx, My = 15, 10
Nx, Ny = 6, 4
nfft2d = nfft.NFFT2D(Mx * My, Nx, Ny)
x = np.linspace(-0.5, 0.5, Mx, endpoint=False)
y = np.linspace(-0.5, 0.5, My, endpoint=False)
xv, yv = np.meshgrid(x, y)
xy = np.zeros([Mx * My, 2])
for i in range(Mx * My):
    xy[i, 0] = xv.ravel()[i]
    xy[i, 1] = yv.ravel()[i]
nfft2d.x = xy
nfft2d.f_hat[:] = np.zeros([Nx, Ny])
nfft2d.f_hat[3, 3] = 1
print(nfft2d.f_hat)
nfft2d.precompute_x()
nfft2d.trafo()
f = nfft2d.f.reshape(My, Mx)
plt.figure(1)
plt.imshow(np.real(f))

# %%

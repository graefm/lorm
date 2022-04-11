import nfft
import numpy as np
import matplotlib.pyplot as plt
%matplotlib

Mp1, Mt, Mp2 = 20, 21, 20
N = 3
nfsoft_plan = nfft.nfsoft.plan(Mp1 * Mt * Mp2, N)

p1 = 2 * np.pi * np.linspace(-0.5, 0.5, Mp1, endpoint=False)
t = 2 * np.pi * np.linspace(0 + 0.00001, 0.5 - 0.00001, Mt, endpoint=True)
p2 = 2 * np.pi * np.linspace(-0.5, 0.5, Mp2, endpoint=False)
tv, p1v, p2v = np.meshgrid(t, p1, p2)
p1tp2 = np.zeros([Mp1 * Mt * Mp2, 3])
for i in range(Mp1 * Mt * Mp2):
    p1tp2[i, 0] = p1v.ravel()[i]
    p1tp2[i, 1] = tv.ravel()[i]
    p1tp2[i, 2] = p2v.ravel()[i]
nfsoft_plan.set_local_coords(p1tp2)

fhat = nfft.nfsoft.SO3FourierCoefficients(N)
fhat[1, 1, -1] = 1

f = nfsoft_plan.compute_Ymatrix_multiplication(fhat)
fhat = nfsoft_plan._get_fhat(N)

p1tp2 = p1tp2.ravel().reshape(Mp1, Mt, Mp2, 3)

f = f.reshape(Mp1, Mt, Mp2)
plt.figure(1)
plt.imshow(np.real(f[:, 5, :]), aspect=Mp2 / Mp1)
plt.figure(2)
plt.plot(np.real(f[int(Mp1 / 2), :, int(Mp2 / 2)]))

# %%
gradf = nfsoft_plan.compute_gradYmatrix_multiplication(fhat)
dphi1 = gradf[:, 0]
dphi1 = dphi1.reshape(Mp1, Mt, Mp2)
plt.figure(1)
plt.imshow(np.real(dphi1[:, 5, :]), aspect=Mp2 / Mp1)
dphi2 = gradf[:, 2]
dphi2 = dphi2.reshape(Mp1, Mt, Mp2)
plt.figure(2)
plt.imshow(np.real(dphi2[:, 5, :]), aspect=Mp2 / Mp1)
dtheta = gradf[:, 1]
dtheta = dtheta.reshape(Mp1, Mt, Mp2)
plt.figure(3)
plt.plot(np.real(dtheta[int(Mp1 / 2), :, int(Mp2 / 2)]))

# %%
fhat = nfsoft_plan.compute_gradYmatrix_adjoint_multiplication(gradf)
for n in range(N + 1):
    print(np.imag(fhat[n, :, :]))

# %%

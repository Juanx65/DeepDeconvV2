# Reject modernity, return to MONKEy

from ISTA import FISTA
import numpy as np
import h5py
from matplotlib import pyplot as plt

kernel = np.flip(np.load("kernels/chirp_kernel.npy"))
kernel /= np.abs(kernel).max()


ISTA_solver = FISTA(kernel=kernel, lam=0.02)
with h5py.File('data/CHIRP_DAS_data.h5', "r") as f:
    _, Nt = f["strainrate"].shape
    data = f["strainrate"][:,::].astype(np.float32)


data /= np.abs(data).max()

print(data.shape)

data0 = np.reshape(data[0,:], (1,1024))
loss, x_hat_fista, y_hat_fista = ISTA_solver.solve(data0, N=10)
print("LOSS :{}".format(loss))
fig, axs = plt.subplots(ncols=1, nrows=4)


# impulses_ISTA = np.zeros((10, 1024))
# for i in range(10):
#     # Select chunk
#     y = np.reshape(data[i, :], (1,1024))
#     loss, x, _ = ISTA_solver.solve(y, N=500)
#     print(i, loss)
#     # Store impulse model
#     impulses_ISTA[i, :] = x 

# NADA FISTA NO HACE NADA


# x_hat_fista /= np.abs(x_hat_fista).max()
# y_hat_fista /= np.abs(y_hat_fista).max()


""" PLOT """
axs[0].plot(data[0,:])
axs[0].set_title('Chirp Data')

axs[1].plot(x_hat_fista)
axs[1].set_title('FISTA X HAT')

axs[2].plot(y_hat_fista)
axs[2].set_title('FISTA Y HAT')

axs[3].plot(kernel)
axs[3].set_title('Kernel')
for ax in axs:
    ax.grid()

plt.show()


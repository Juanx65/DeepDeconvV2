from matplotlib import pyplot as plt
import os
import numpy as np

kernel_dir = '../kernels'
kernels_names = os.listdir(kernel_dir)

names = 'abcdefghijklmnopqrstuvwxyz'

fig, axs = plt.subplots(ncols=1, nrows=len(kernels_names))

for i, kernel in enumerate(kernels_names):
    k = np.load(os.path.join(kernel_dir, kernel))
    axs[i].plot(k)
    axs[i].set_title(kernel)
fig.tight_layout()

plt.show()
import os
import numpy as np
import scipy.fft
from scipy.signal import windows
import matplotlib.pyplot as plt
import h5py
from ISTA import FISTA
from time import time
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt


letter_params = {
    "fontsize": 10,
    "verticalalignment": "top",
    "horizontalalignment": "left",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

cwd = os.getcwd()
datadir = os.path.join(cwd, "data")
data_file = os.path.join(datadir, "DAS_data.h5")

gauge = 3.2
samp = 50.
v_ref = 90. / 3.6

buf = 100_000

""" Load DAS data """

with h5py.File(data_file, "r") as f:
    Nch, Nt = f["strainrate"].shape
    split = int(0.9 * Nt)
    data = f["strainrate"][:, split:-buf].astype(np.float32)
    
data /= data.std()
Nch, Nt = data.shape


""" Integrate DAS data (strain rate -> strain) """  

win = windows.tukey(Nt, alpha=0.1)
freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
Y = scipy.fft.rfft(win * data, axis=1)
Y_int = -Y / (2j * np.pi * freqs)
Y_int[:, 0] = 0
data_int = scipy.fft.irfft(Y_int, axis=1)
data_int /= data_int.std()


""" Load impulse response """

kernel = np.load(os.path.join(datadir, "kernel.npy"))
kernel = kernel / kernel.max()


""" Integrate impulse response (strain rate -> strain) """

win = windows.tukey(len(kernel), alpha=0.9)
freqs = scipy.fft.rfftfreq(len(kernel), d=1/samp)
Y = scipy.fft.rfft(win * kernel)
Y_int = -Y / (2j * np.pi * freqs)
Y_int[0] = 0

kernel_int = scipy.fft.irfft(Y_int) + 0.01
kernel_int = kernel_int * win
kernel_int /= np.abs(kernel_int).max()
kernel_int = kernel_int.astype(np.float32)

"""Process with FISTA""" 
# Number of time samples per chunk
ISTA_win = 5_000
# Number of chunks
Nwin = (Nt - ISTA_win) // ISTA_win
# Total number of samples to be processed
Nt_ISTA = Nwin * ISTA_win

# Buffer for impulses
impulses_ISTA = np.zeros((Nch, Nt_ISTA))

# Init solver
ISTA_solver = FISTA(kernel=kernel, lam=0.02)

t0 = time()

# Loop over time chunksDone 3790.2608981132507clea

for i in range(Nwin):
    
    # Select chunk
    t_slice = slice(i * ISTA_win, (i + 1) * ISTA_win)
    y = data[:, t_slice]
    
    # Run FISTA with 50 iterations
    loss, x, _ = ISTA_solver.solve(y, N=50)
    
    print(i, loss)
    
    # Store impulse model
    impulses_ISTA[:, t_slice] = x

t1 = time()
    
data    = asarray(impulses_ISTA)
savetxt('impulse_ISTA.csv', data, delimiter=',')


arch_time = open("time_log.txt","w")
time_diff = t1-t0


arch_time.write(str(time_diff)+ "\n")
arch_time.close()

print("Done", time_diff)


"""Deconvolution results"""
""" Some interesting examples to plot  """

examples = {
    "light1": {
        "slice": slice(98_000, 100_000),
    },
    "light2": {
        "slice": slice(33_000, 39_000),
    },
    "truck": {
        "slice": slice(219_900, 221_250),
    },
    "heavy1": {
        "slice": slice(391_800, 393_000),
    },
    "heavy2": {
        "slice": slice(388_600, 389_800),
    },
    "heavy3": {
        "slice": slice(511_850, 513_400),
    }
}

"""plots"""
scale = 0.02  # spacing between wiggles

# Draw canvas
plt.close("all")
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(9, 6), constrained_layout=True, sharex="col", sharey="row")

# Remove spines from all panels
for ax in axes.ravel():
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

# Loop over examples
for j, example in enumerate((examples["light1"], examples["heavy1"])):
    
    # Time vector
    t = np.arange(example["slice"].stop - example["slice"].start) / samp
    
    # Set x-axis limits
    for ax in axes[:, j]:
        ax.set_xlim((t.min(), t.max()))
    
    # Plot examples
    ax = axes[0, j]
    for i, wv in enumerate(data[:, example["slice"]]):
        ax.plot(t, wv - 2 * i, c="k")
    ax = axes[1, j]
    for i, wv in enumerate(impulses_ISTA[:, example["slice"]]):
        ax.plot(t, wv - scale * i, c="k")
    ax = axes[2, j]
    for i, wv in enumerate(impulses_ISTA[:, example["slice"]]):
        ax.plot(t, wv - scale * i, c="k")
        
# Set x-label
for ax in axes[-1]:
    ax.set_xlabel("time [s]")
    
# Set y-labels
axes[0, 0].set_ylabel("Original", fontsize=14, labelpad=12)
axes[1, 0].set_ylabel("FISTA", fontsize=14, labelpad=12)
axes[2, 0].set_ylabel("FISTA", fontsize=14, labelpad=12)

# Add panel letters
for ax, letter in zip(axes.ravel(), "abcdefghijk"):
    ax.text(x=0.01, y=0.93, s=letter, transform=ax.transAxes, **letter_params)
        
plt.savefig("examples_deconv.pdf")
plt.show()

data = loadtxt('impulse_ISTA.csv', delimiter=',')
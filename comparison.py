import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
from pathlib import Path
import scipy.fft
from scipy.signal import windows
from utils.models import UNet
import tensorflow as tf
from time import time

""" LETTER PARAMS """
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

""" LOAD DATA """
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

###############################################################################
impulses_ISTA = np.loadtxt('impulse_ISTA.csv', delimiter=',')
###############################################################################

""" DEEP DECONVOLUTION """
rho = 10.0
f0 = 8
blocks = 3
noise = 1.0
# Number of samples per chunk
deep_win = 1024
# Number of chunks
Nwin = data.shape[1] // deep_win
# Total number of time samples to be processed
Nt_deep = Nwin * deep_win
# Define a dummy
dummy = tf.zeros((2, Nch, deep_win, 1))
# Location of pretrained model
cwd = os.getcwd()
datadir = os.path.join(cwd, "weights")
loadfile = os.path.join(datadir, "autores.h5")
# Init Deep Learning model
model = UNet(
    kernel_int, lam=rho, f0=f0, 
    data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise, causal=False,activation_function="relu"
)
model.construct()
# Someone at Google HQ decided it would be good to do some voodoo first
# Calling the model is needed in order to load the weights
_ = model(dummy)
# Load pretrained weights
model.load_weights(loadfile)



""" Mould data into right shape for UNet """
data_split = np.stack(np.split(data_int[:, :Nt_deep], Nwin, axis=-1), axis=0)
data_split = np.stack(data_split, axis=0)
data_split = np.expand_dims(data_split, axis=-1)

# Buffer for impulses
x = np.zeros_like(data_split)
y = np.zeros_like(data_split)

batch_size = 32
N = data_split.shape[0] // batch_size
r = data_split.shape[0] % batch_size

t0 = time()

# Loop over chunks
for i in range(data_split.shape[0] // batch_size):
    print(i)
    n_slice = slice(i * batch_size, (i + 1) * batch_size)
    x_i, y_i = model(data_split[n_slice])
    x[n_slice] = x_i
    y[n_slice] = y_i
    
# If there is some residual chunk: process that too
if r > 0:
    n_slice = slice((i + 1) * batch_size, None)
    x_i, y_i = model(data_split[n_slice])
    x[n_slice] = x_i
    y[n_slice] = y_i

impulses_deep = np.concatenate(np.squeeze(x), axis=1)
y_hat_authors_total = np.concatenate(np.squeeze(y), axis=1)

t1 = time()

print("Done", t1 - t0)
###############################################################################
###############################################################################
""" OUR WEIGHTS"""
""" Init Deep Learning model """
our_model = UNet(
    kernel.astype(np.float32), lam=rho, f0=f0,
    data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise, causal=False, activation_function="relu"
)

our_model.construct()
our_model.compile()

""" CARGAR PESOS AL MODELO """
our_model.load_weights(str(str(Path(__file__).parent) + '/weights/1000-epoch-authors-integrado/best.ckpt')).expect_partial()#'/checkpoints/cp-0100.ckpt'))

""" Mould data into right shape for UNet """
data_split = np.stack(np.split(data_int[:, :Nt_deep], Nwin, axis=-1), axis=0)
data_split = np.stack(data_split, axis=0)
data_split = np.expand_dims(data_split, axis=-1)

# Buffer for impulses
x = np.zeros_like(data_split)
y = np.zeros_like(data_split)

batch_size = 32
N = data_split.shape[0] // batch_size
r = data_split.shape[0] % batch_size

t0 = time()

# Loop over chunks
for i in range(data_split.shape[0] // batch_size):
    print(i)
    n_slice = slice(i * batch_size, (i + 1) * batch_size)
    x_i, y_i = our_model(data_split[n_slice])
    x[n_slice] = x_i
    y[n_slice] = y_i
    
# If there is some residual chunk: process that too
if r > 0:
    n_slice = slice((i + 1) * batch_size, None)
    x_i, y_i = our_model(data_split[n_slice])
    x[n_slice] = x_i
    y[n_slice] = y_i

impulses_deep_ours = np.concatenate(np.squeeze(x), axis=1)
y_hat_ours_total = np.concatenate(np.squeeze(y), axis=1)

t1 = time()

print("Done", t1 - t0)








"""Deconvolution results"""
""" Some interesting examples to plot  """

window = 1024
examples = {
    "light1": {
        "slice": slice(30_000, 40_000),
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
    },
    "light1024": {
        "slice": slice(219_900, 219_900 + window - 1)
    },
    "heavy1024": {
        "slice": slice(400_000, 500_000)
    }
}

"""plots"""
scale = 0.02  # spacing between wiggles
scale_height = [0.3, 1]
scale_height2 = [0.14, 1]
samp = 50.
# Draw canvas
plt.close("all")
fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(18, 6), constrained_layout=True, sharex="col", sharey="row")

# Remove spines from all panels
for ax in axes.ravel():
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

# Loop over examples
for j, example in enumerate((examples["heavy3"], examples["heavy2"])):
    
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
        ax.plot(t, scale_height[j]*wv - scale * i, c="k")
    ax = axes[2, j]
    for i, wv in enumerate(impulses_deep[:, example["slice"]]):
        ax.plot(t, scale_height[j]*wv - scale * i, c="k")
    ax = axes[3, j]
    for i, wv in enumerate(y_hat_authors_total[:, example["slice"]]):
        ax.plot(t, wv - 2 * i, c="k")
    ax = axes[4, j]
    for i, wv in enumerate(impulses_deep_ours[:, example["slice"]]):
        ax.plot(t, scale_height2[j]*wv - scale * i, c="k")
    ax = axes[5, j]
    for i, wv in enumerate(y_hat_ours_total[:, example["slice"]]):
        ax.plot(t, wv - 2 * i, c="k")

        
        
# Set x-label
for ax in axes[-1]:
    ax.set_xlabel("time [s]")
    
# Set y-labels
axes[0, 0].set_ylabel("Original", fontsize=14, labelpad=12)
axes[1, 0].set_ylabel("FISTA", fontsize=14, labelpad=12)
axes[2, 0].set_ylabel("Impulses\nDAE\n(authors)", fontsize=14, labelpad=12)
axes[3, 0].set_ylabel("Reconstruction\nDAE\n(authors)", fontsize=14, labelpad=12)
axes[4, 0].set_ylabel("Impulses\nDAE\n(ours)", fontsize=14, labelpad=12)
axes[5, 0].set_ylabel("Reconstruction\nDAE\n(ours)", fontsize=14, labelpad=12)


# Add panel letters
for ax, letter in zip(axes.ravel(), "abcdefghijk"):
    ax.text(x=0.01, y=0.93, s=letter, transform=ax.transAxes, **letter_params)
        
plt.savefig("examples_deconv.pdf")
plt.show()
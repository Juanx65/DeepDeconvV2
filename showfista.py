import numpy as np
from matplotlib import pyplot as plt
import h5py
import os

""" Load DAS data """
cwd = os.getcwd()
datadir = os.path.join(cwd, "data")
data_file = os.path.join(datadir, "DAS_data.h5")
buf = 100_000
with h5py.File(data_file, "r") as f:
    Nch, Nt = f["strainrate"].shape
    split = int(0.9 * Nt)
    data = f["strainrate"][:, split:-buf].astype(np.float32)
    
data /= data.std()
#Nch, Nt = data.shape


impulses_ISTA = np.loadtxt('impulse_ISTA.csv', delimiter=',')

letter_params = {
    "fontsize": 10,
    "verticalalignment": "top",
    "horizontalalignment": "left",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}


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
        "slice": slice(388_600, 388_600 + window - 1)
    }
}

"""plots"""
scale = 1.5 # spacing between wiggles
samp = 50.
# Draw canvas
plt.close("all")
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(9, 6), constrained_layout=True, sharex="col", sharey="row")

# Remove spines from all panels
for ax in axes.ravel():
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

# Loop over examples
for j, example in enumerate((examples["light1024"], examples["heavy3"])):
    
    # Time vector
    t = np.arange(example["slice"].stop - example["slice"].start) / samp
    
    # Set x-axis limits
    for ax in axes[:, j]:
        ax.set_xlim((t.min(), t.max()))
    
    # Plot examples
    ax = axes[0, j]
    for i, wv in enumerate(data[:, example["slice"]]):
        ax.plot(t, wv - i, c="k")
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
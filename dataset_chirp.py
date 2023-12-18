import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, convolve

def fade(signal, fs, T, window_func):
    L = len(signal)
    N = np.round(np.array(T) * 1e-3 * fs).astype(int)
    fadein = fade_window(N[0], 'fadein', window_func, signal.ndim)
    signal[:N[0]] *= fadein
    fadeout = fade_window(N[1], 'fadeout', window_func, signal.ndim)
    signal[-N[1]:] *= fadeout
    return signal

def fade_window(N, fade_type, custom_func, dims):
    window = custom_func(2 * N)
    window = window[:N]
    window /= np.max(np.abs(window))
    if fade_type in ['fadeout', 'fade-out']:
        window = window[::-1]
    return window.reshape((-1, 1)) if dims == 2 else window

# Parámetros
n = 1024
kn = n // 2
maxt = 2

t = np.linspace(0, maxt, kn)
chirp_kernel = chirp(t, 0.1, maxt, 20)
chirp_kernel = fade(chirp_kernel, kn / maxt, [400, 400], lambda N: np.hanning(N))

impulse_rnd = np.zeros(n)
q = np.random.permutation(n)
for i in range(5):
    impulse_rnd[q[i]] = 2 * np.random.rand() - 1

t2 = np.linspace(0, 2 * maxt, n)

output = convolve(impulse_rnd, chirp_kernel, mode='full')
output = output[:1024]

# Visualización
plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(t, chirp_kernel, linewidth=1.2)
plt.subplot(3, 1, 2)
plt.stem(t2, impulse_rnd, linefmt='-', markerfmt='o', basefmt=' ')
plt.ylim([-1, 1])
plt.subplot(3, 1, 3)
plt.plot(t2, output, linewidth=1.2)
plt.ylim([-2, 2])
plt.tight_layout()
plt.show()

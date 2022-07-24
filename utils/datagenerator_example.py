import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.fft
from scipy.signal import windows
from models import DataGenerator

# Load Data
data_file = 'DAS_data.h5'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='Show plots')

    args = parser.parse_args()
    return args

args = parse_opt()

# Constants
buf = 100_000
samp = 50.
deep_win = 1024

with h5py.File(data_file, 'r') as file:
    print("El formato extraido del archivo : {}".format(file['strainrate'].shape))
    Nch, Nt = file['strainrate'].shape
    split = int(0.9 * Nt)
    print("Nch {}, Nt {}, split {}".format(Nch, Nt, split))
    print(file["strainrate"][23].shape)
    # La data cargada son 24 canales registrados sobre el intervalo total.
    data = file["strainrate"][:, split:-buf].astype(np.float32)
    # En la linea anterior, de todos los canales a partir del 0.9 de las muestras totales, se escoge un rango desde 7.763.490 hasta 8.626.100 - 100.000 = 8.526.100. Lo que da un total de 762.610 samples
    print("data tiene la siguiente forma {}".format(data.shape))
    if args.show:
        plt.plot(file['strainrate'][0,split:-buf])
        plt.grid()

    # Normalizar datos
    data /= data.std()
    Nch, Nt = data.shape
    print("Data recortado - Nch {}, Nt {}".format(Nch, Nt))


    # Integrar datas
    win = windows.tukey(Nt, alpha = 0.1)
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
    Y = scipy.fft.rfft(win*data, axis=1)
    Y_int = -Y/(2j* np.pi * freqs)
    Y_int[:,0] = 0
    data_int = scipy.fft.irfft(Y_int, axis=1)
    data_int /= data_int.std()

    if args.show:
        plt.plot(data_int[0,:], c='r')
        plt.show()


    # Generar dataset
    DataGenerator(data_int, 1024,10000, 128)

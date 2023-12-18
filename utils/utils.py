import os
import numpy as np
import h5py
from scipy.io import loadmat
import scipy.integrate as it
import scipy.fft
from scipy.signal import windows
from random import randint
# Model constants
# -> Data constants
buf = 100_000
samp = 50. #sampling time

# -> Model parameters [TO-DO]: This is from the paper
rho = 10.0
f0 = 8
blocks = 3



# Converts data from .mat file to h5py
def dataConverter(file_path, outfile_path, name, key='array_output'):
    if not(os.path.exists(file_path)):
        print("Data file does not exists!")
        exit(1)
    try:
        # Read mat data to np array
        raw_mat = loadmat(file_path)
        mat_data = np.array(raw_mat[key])
        with h5py.File(outfile_path, "w") as f:
            f.create_dataset(name, data=mat_data)
        print('File {} was converted to h5py: {}'.format(file_path, outfile_path))
    except Exception as e:
        print('Something went wrong ...')
        print(e)

# Integrate DAS data: Performs the integration of data in the frequency domain
# data = data to integrate
# samp = sampling time
def integrateDAS(data):
    _, Nt = data.shape
    win = windows.tukey(Nt, alpha=0.1)
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
    Y = scipy.fft.rfft(win * data, axis=1)
    Y_int = -Y / (2j * np.pi * freqs)
    Y_int[:, 0] = 0
    data_int = scipy.fft.irfft(Y_int, axis=1)
    data_int /= data_int.std()
    return data_int


## desfase de data para cada canal de manera circular
## lista: dato Original
## idx: indice a partir de donde se copia del primer dato
def fill_channel(lista,idx):
    channel =np.zeros(len(lista))

    for i  in range(len(lista[idx:])):
        channel[idx + i] = lista[i]


    return channel
# Augmentate the simulated dataset
def moreChannels(file_path, outfile_path, key='strainrate', phase=False):
    if not(os.path.exists(file_path)):
        print("Data file does not exists!")
        exit(1)
    try:
        # Read single channel data
        with h5py.File(file_path, 'r') as f:
            new_data = []
            for dato in f[key]:
                phase_val = randint(0,8) if phase else 0
                temp_dato = np.zeros((24,1024))
                # first channel
                temp_dato[0] = dato
                for ch in range(1,23):
                    dato  = fill_channel(dato, (ch+1)*phase_val)
                    temp_dato[ch] = dato
                new_data.append(temp_dato)
        new_data = np.array(new_data)
        print(new_data.shape)
        # Write multichannel data
        with h5py.File(outfile_path, 'w') as f:
            f.create_dataset(key, data=new_data)
            print('{} was sucessfully writen!'.format(outfile_path))

    except Exception as e:
        print('Something went wrong ...')
        print(e)

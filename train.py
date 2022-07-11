import os
from pathlib import Path
import numpy as np
import scipy.fft
from scipy.signal import windows
import matplotlib.pyplot as plt
import h5py
from models import UNet
from models import CallBacks
from random import choice
import tensorflow as tf
import argparse
from models import DataGenerator
from datetime import datetime
import json
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

def train(opt):

    """ Variables necesarias """
    cwd = os.getcwd()
    datadir = os.path.join(cwd, opt.data_dir)
    kerneldir =  os.path.join(cwd, opt.kernel_dir)
    data_file = os.path.join(datadir, "DAS_data.h5")

    samp = 50.

    epochs = opt.epochs
    batch_size = opt.batch_size

    """ Load DAS data """
    # DAS_data.h5 -> datos para leer (1.5GB) strain rate -> hay que integrarlos
    with h5py.File(data_file, "r") as f:
         # Nch : numero de canales, Nt = largo de muestras 8.626.100
        Nch, Nt = f["strainrate"].shape
        split = int(0.9 * Nt) #90% datos para entrenamiento y validación
        data = f["strainrate"][:, 0:split].astype(np.float32)
    # se normaliza cada trace respecto a su desviación estandar
    data /= data.std()
    Nch, Nt = data.shape
    # Shape: 24 x 180_000 (son 24 sensores/canales, y 180_000 muestras?)


    """ Integrate DAS data (strain rate -> strain) """
    win = windows.tukey(Nt, alpha=0.1)
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
    Y = scipy.fft.rfft(win * data, axis=1)
    Y_int = -Y / (2j * np.pi * freqs)
    Y_int[:, 0] = 0
    data_int = scipy.fft.irfft(Y_int, axis=1)
    data_int /= data_int.std()

    """ Call DataGenerator """
    window = opt.deep_win
    samples_per_epoch = 1000 # data que se espera por epoca al entrenar
    batches = opt.batch_size
    train_val_ratio = 0.5
    _, Nt_int = data_int.shape
    split = int(0.5 * Nt_int)

    train_raw_data = data_int[:,0:split]
    val_raw_data = data_int[:,split:]

    train_data = DataGenerator(train_raw_data, window, samples_per_epoch, batches)
    val_data = DataGenerator(val_raw_data, window, samples_per_epoch, batches)
    ########################################3
    """ Load impulse response """
    #kernel = np.load(os.path.join(datadir, "kernel.npy"))
    kernel = np.load(os.path.join(kerneldir,"i_kernel.npy")) # integrado
    # Se normaliza el kernel respecto al máximo (a diferencia de las traces DAS que se normalizan respecto a la desviación estandar)
    kernel = kernel / kernel.max()

    """ Some model parameters """
    rho = 10.0
    f0 = 8
    blocks = 3
    dropout_value = opt.dropout
    deep_win = opt.deep_win

    """ Init Deep Learning model """
    model = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=dropout_value
    )

    model.construct()
    model.compile()

    checkpoint_filepath = str(str(Path(__file__).parent) +opt.checkpoint)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True,
        update_freq="epoch")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[model_checkpoint_callback],
        batch_size=batch_size
    )


    timeID = datetime.now()
    timeString = timeID.strftime("%Y-%m-%d_%H-%M-%S")
    file_prefix = 'train_{}.json'.format(timeString)

    with open(os.path.join('trainHistory/',file_prefix), 'w') as file:
        json.dump(history.history, file)

    """ Printear algunas cosas para ver como se entreno """
    loss1 = history.history['l1']
    val_loss1 = history.history['val_l1']

    loss2 = history.history['l2']
    val_loss2 = history.history['val_l2']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)#epochs

    plt.figure(figsize=(8, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, loss1, label='Training Sparsity (L1)')
    plt.plot(epochs_range, val_loss1, label='Validation Sparsity (L1)')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Sparsity')

    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Total Loss')

    plt.subplot(3, 1, 3)
    plt.plot(epochs_range, loss2, label='Training Loss (L2)')
    plt.plot(epochs_range, val_loss2, label='Validation Loss (L2)')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss (L2)')

    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 128, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 200 ,type=int,help='epoch to train')
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--kernel_dir', default = "kernels",type=str,help='dir to the dataset')
    parser.add_argument('--checkpoint', default = "/checkpoints/best.ckpt",type=str,help='dir to save the weights og the training')
    parser.add_argument('--dropout', default = 1.0,type=float,help='percentage dropout to use')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')

    opt = parser.parse_args()
    return opt

def main(opt):
    train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

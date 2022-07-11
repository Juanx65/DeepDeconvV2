####################################################################################################
                          """IMPORTS SECTION"""
####################################################################################################
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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'


####################################################################################################
                            """ TEST FUNCTION """
####################################################################################################
def test(opt):
    cwd = os.getcwd()                                 #NO SE
    datadir = os.path.join(cwd, opt.data_dir)
    kerneldir =  os.path.join(cwd, opt.kernel_dir)
    data_file = os.path.join(datadir, "DAS_data.h5")
    buf = 100_000
    samp = 50.


    """ CARGAR EL MODELO """
    """ Load impulse response """
    kernel = np.load(os.path.join(kerneldir,"i_kernel.npy"))
    # Se normaliza el kernel respecto al máximo (a diferencia de las traces DAS que se normalizan respecto a la desviación estandar)
    kernel = kernel / kernel.max()

    """ Some model parameters """
    rho = 10.0
    f0 = 8
    blocks = 3
    noise = opt.dropout
    deep_win = opt.deep_win

    """ CARGAR DATA PARA PRUEBAS """

    """ Load DAS data """
    # DAS_data.h5 -> datos para leer (1.5GB) strain rate -> hay que integrarlos
    with h5py.File(data_file, "r") as f:
         # Nch : numero de canales, Nt = largo de muestras (1024?), SI
        Nch, Nt = f["strainrate"].shape
        split = int(0.45 * Nt) #incluye todo menos train data
        data = f["strainrate"][:, split:].astype(np.float32)
    # se normaliza cada trace respecto a su desviación estandar
    data /= data.std()
    Nch, Nt = data.shape

    """ Init Deep Learning model """
    model = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise
    )

    model.construct()
    model.compile()

    """ CARGAR PESOS AL MODELO """
    model.load_weights(str(str(Path(__file__).parent) + opt.weights)).expect_partial()#'/checkpoints/cp-0100.ckpt'))

    """ Integrate DAS data (strain rate -> strain) """

    win = windows.tukey(Nt, alpha=0.1)
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)
    Y = scipy.fft.rfft(win * data, axis=1)
    Y_int = -Y / (2j * np.pi * freqs)
    Y_int[:, 0] = 0
    data_int = scipy.fft.irfft(Y_int, axis=1)
    data_int /= data_int.std()

    ########################################################

    Nwin = data_int.shape[1] // deep_win
    # Total number of time samples to be processed
    Nt_deep = Nwin * deep_win #
    #
    data_split = np.stack(np.split(data_int[:, :Nt_deep], Nwin, axis=-1), axis=0)
    data_split = np.stack(data_split, axis=0)
    data_split = np.expand_dims(data_split, axis=-1)
    # Buffer for impulses
    batch_size = 1 # PARA TENER SOLO UN DATO EN 1 BATCH

    x = np.zeros_like(data_split)
    N = data_split.shape[0] // batch_size
    r = data_split.shape[0] % batch_size
    for i in range(N):
        n_slice = slice(i * batch_size, (i + 1) * batch_size)
        x_i = data_split[n_slice]
        x[n_slice] = x_i
    # If there is some residual chunk: process that too
    if r > 0:
        n_slice = slice((N-1 + 1) * batch_size, None)
        x_i = data_split[n_slice]
        x[n_slice] = x_i

    """ FINALMENTE HACER LA PRUEBA"""
    good_datas = open("datos_buenos.txt","w")
    for i in range(len(x)):
        suma = sum(abs(x[i]))
        suma_tot = [o[0] for o in suma]
        suma_tot = sum(suma_tot)

        if suma_tot > 25000: # para considerar datos en los que ocurra algo significativo
            good_datas.write(str(i)+","+str(suma_tot)+"\n")

    good_datas.close()

    i = input("index data show: ")
    while True:
        if i != "":
            image_index = int(i)
            x_hat, y_hat = model.call(x[image_index][None,:,:,:])
            x_hat = tf.reshape(x_hat,[24,1024])
            y_hat = tf.reshape(y_hat,[24,1024])

            """ GRAFICAR LOS RESULTADOS """
            samp = 80.
            t = np.arange(x_hat.shape[1]) / samp


            f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
            ax1.set_title('S')
            ax2.set_title('E_hat')
            ax3.set_title('S_hat')

            f.suptitle('DATA'+ str(i), fontsize=16)
            #subplot1: origina
            for i, wv in enumerate(x[image_index]):
                ax1.plot( t, wv - 8 * i, "tab:orange",linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #subplot2: x_hat-> estimación de la entrada (conv kernel con la salida)
            for i, wv in enumerate(x_hat):
                ax2.plot(t,(10*wv - 8 * i), "tab:red", linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #subplot3: y_hat->
            for i, wv in enumerate(y_hat):
                ax3.plot(t,wv - 8 * i, c="k",linewidth=2.5)
            plt.tight_layout()
            plt.grid()

            #plt.savefig("figures/multi_cars_impulse.pdf")
            plt.grid()
            plt.show()
            plt.close()
        else:
            break
        i = input("index data show: ")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = "data",type=str,help='dir to the dataset')
    parser.add_argument('--kernel_dir', default = "kernels",type=str,help='dir to the dataset')
    parser.add_argument('--weights',default = '/checkpoints/best.ckpt', type=str,help='load weights path')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='optimizer for the model ej: adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 1.0,type=float,help='% dropout to use')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
	test(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

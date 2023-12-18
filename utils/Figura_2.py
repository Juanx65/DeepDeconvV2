################################################################################
""" IMPORTS SECTION """
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
from models import UNet
from models import CallBacks
from random import choice
import tensorflow as tf
import argparse
from utils import *

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

################################################################################
# Global consts
default_kernel = 'kernels/chirp_kernel.npy' # For simulated chirped data
default_data = 'data/CHIRP_DAS_NOFASE_data.h5'
authors_data = 'data/DAS_data.h5'
################################################################################
""" TEST FUNCTION """
################################################################################
def test(opt):
    noise = opt.dropout
    deep_win = opt.deep_win

    """ LOAD KERNEL """
    # Verify if file exists
    if not(os.path.exists(opt.kernel)):
        print("The kernel file <{}> does not exists!".format(opt.kernel))
        exit(1)
    kernel = np.load(opt.kernel)
    if(opt.integrate):
        kernel = it.cumtrapz(kernel, np.linspace(0,1,len(kernel)), initial=0.0)
    if not (opt.perform_crosscorrelation):
        kernel = np.flip(kernel)

    kernel = kernel / kernel.max() # Kernel normalization


    """ CARGAR DATA PARA PRUEBAS """

    """ Load DAS data """
    if(opt.authors):
        opt.data = authors_data
        print("--authors arg passed. Data {} will be ignored!".format(opt.data))

    if not(os.path.exists(opt.data)):
        print("Data file {} does not exists!".format(opt.data))
        exit(1)
    with h5py.File(opt.data, "r") as f:
        # Nch : numero de canales (24), Nt = largo de muestras
        if(opt.authors):
            Nch, Nt = f["strainrate"].shape
            split = int(0.45 * Nt) #incluye todo menos train data
            data = f["strainrate"][:, split:].astype(np.float32)
        else:
            _,Nch, Nt = f["strainrate"].shape
            data = f["strainrate"][:,::].astype(np.float32)

    # se normaliza cada trace respecto a su desviación estandar
    data /= data.std()

    """ Integrate DAS data """
    # The original work integrates the data, so it's left as an option, but not in use anymore.
    if (opt.integrate):
        data = integrateDAS(data)

    """ Init Deep Learning model: Cross-Correlation """
    model_cc = UNet(
        np.flip(kernel).astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise, causal=opt.noncausal_convolution, activation_function=opt.act_function
    )
    model_cc.construct()
    model_cc.compile()

    """ Init Deep Learning model: Convolution (Invertion of kernel) """
    model_conv = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise, causal=opt.noncausal_convolution, activation_function=opt.act_function
    )
    model_conv.construct()
    model_conv.compile()

    """ CARGAR PESOS AL MODELO """
    model_cc.load_weights(str(str(Path(__file__).parent.parent) + '/weights/200-epoch-chirp-noflip/best.ckpt')).expect_partial()
    model_conv.load_weights(str(str(Path(__file__).parent.parent) + '/weights/200-epoch-chirp-single-channel/best.ckpt')).expect_partial()

    Nwin = data.shape[1] // deep_win
    # Total number of time samples to be processed
    Nt_deep = Nwin * deep_win #

    #
    if(opt.authors):
        data_split = np.stack(np.split(data[:, :Nt_deep], Nwin, axis=-1), axis=0)
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


    else:
        x = data

    """ LETTER PARAMS """
    letter_params = {
    "fontsize": 10,
    "verticalalignment": "top",
    "horizontalalignment": "left",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
    }  

    # Init solver
   

    i = input("index 1 data show: ")
    while True:
        if i != "":
            # Agregar contador de tiempo para DAE y FISTA
            image_index = int(i)
            if(opt.authors):
                x_hat, y_hat = model_cc.call(x[image_index][None,:,:,:])
                x_hat1, y_hat1 = model_conv.call(x[image_index][None,:,:,:])
            else:
                x_hat, y_hat = model_cc.call(x[image_index][None,:,:])
                x_hat1, y_hat1 = model_conv.call(x[image_index][None,:,:])

            x_hat = tf.reshape(x_hat,[24,1024])
            y_hat = tf.reshape(y_hat,[24,1024])
            x_hat1 = tf.reshape(x_hat1,[24,1024])
            y_hat1 = tf.reshape(y_hat1,[24,1024])

            """ PLOTS """
            samp = 50.
            scale = 2
            # Draw canvas
            plt.close("all")
            fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(9, 6), constrained_layout=True, sharex="col", sharey="row")

            # Remove spines from all panels
            for ax in axes.ravel():
                ax.set_yticks([])
                for spine in ("left", "right", "top"):
                    ax.spines[spine].set_visible(False)

            t = np.arange(x_hat.shape[1]) / samp

            """ Embrace programacion de simio """
            # Set x-axis limits
            #for ax in axes[:, :]:
            #    ax.set_xlim((t.min(), t.max()))

            # Plot examples
            ax = axes[0, 0]
            for i, wv in enumerate(x[image_index]):
                ax.plot(t, wv - scale * i, c="k")
                ax.set_title('Cross-correlation', fontsize=14)
                break
            ax = axes[1, 0]
            for i, wv in enumerate(x_hat):
                ax.plot(t, 0.0001*wv - 100000*scale *i, c="k")
                break
            ax = axes[2, 0]
            for i, wv in enumerate(y_hat):
                ax.plot(t, wv - scale * i, c="k")
                break
            ax = axes[0, 1]
            for i, wv in enumerate(x[image_index]):
                ax.plot(t, wv - scale * i, c="k")
                ax.set_title('Convolution', fontsize=14)
                break
            ax = axes[1, 1]
            for i, wv in enumerate(x_hat1):
                ax.plot(t, 0.0001*wv - 100000*scale *i, c="k")
                break
            ax = axes[2, 1]
            for i, wv in enumerate(y_hat1):
                ax.plot(t, wv - scale * i, c="k")
                break

            # Set x-label
            for ax in axes[-1]:
                ax.set_xlabel("time [s]")
     
            # Set y-labels
            axes[0, 0].set_ylabel("Original", fontsize=14, labelpad=12)
            axes[1, 0].set_ylabel("Impulses", fontsize=14, labelpad=12)
            axes[2, 0].set_ylabel("Reconstruction", fontsize=14, labelpad=12)
   
            # Add panel letters
            for ax, letter in zip(axes.ravel(), "abcdefghijk"):
                ax.text(x=0.01, y=0.93, s=letter, transform=ax.transAxes, **letter_params)
            
            #plt.grid()
            plt.show()
            plt.close()
        else:
            break
        i = input("index 1 data show: ")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noncausal_convolution', action='store_false', help='Determines whether causal or noncausal convolution is used.')
    parser.add_argument('--data', default=default_data ,type=str,help='Dataset to load.')
    parser.add_argument('--act_function', default = "tanh", type=str, help='Activation function for the last layer of UNet e.g. tanh, relu')
    parser.add_argument('--weights',default = '/checkpoints/best.ckpt', type=str,help='Load weights path.')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='Optimizer for the model e.g. adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 1.0,type=float,help='Percentage dropout to use.')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk.')
    parser.add_argument('--integrate', action = 'store_true', help='Indicates if the DAS data and kernel should be integrated.')
    parser.add_argument('--kernel', default = default_kernel, help='Indicates which kernel to use. Receives a <npy> file.')
    parser.add_argument('-pcc','--perform_crosscorrelation', action='store_true', help='Flips kernel in the horizontal axis to perform the cross-correlation. By default perfoms the convolution')
    parser.add_argument('--authors', action='store_true', help='Data from the original work is used, which has a weird shape.')

    opt = parser.parse_args()
    return opt

def main(opt):
	test(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
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
import scipy.fft
from scipy.signal import windows
from ISTA import FISTA
from time import time #time comparison
import argparse
from utils import *

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

################################################################################
# Global consts
#default_kernel = 'kernels/autores_kernel_inv.npy' # Kernel de los autores
default_kernel = 'kernels/kernel.npy' # Kernel de los autores
default_data = 'data/DAS_data.h5'
authors_data = 'data/DAS_data.h5'

def comparison(opt):
    #########################################################################
    """ LOAD KERNEL """
    # Verify if file exists
    if not(os.path.exists(opt.kernel)):
        print("The kernel file <{}> does not exists!".format(opt.kernel))
        exit(1)
    kernel = np.load(opt.kernel)
    if(opt.integrate):
        kernel = it.cumtrapz(kernel, np.linspace(0,1,len(kernel)), initial=0.0)
    if(opt.perform_crosscorrelation):
        kernel = np.flip(kernel)

    kernel = kernel / kernel.max() # Kernel normalization
    #########################################################################
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
    #########################################################################
    """ Integrate DAS data """
    # The original work integrates the data, so is left as an option, but not in use anymore.
    if (opt.integrate):
        data = integrateDAS(data)

    deep_win = opt.deep_win
    # Number of time samples per chunk
    ISTA_win = deep_win #5_000
    # Number of chunks
    Nwin = (Nt - ISTA_win) // ISTA_win
    # Total number of samples to be processed
    Nt_ISTA = Nwin * ISTA_win
    # Buffer for impulses
    impulses_ISTA = np.zeros((Nch, Nt_ISTA))

    # Init solver
    ISTA_solver = FISTA(kernel=kernel, lam=0.02)

    t0 = time()

    # Loop over time chunks
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

    print("Done in ", t1-t0, "seconds")

    """ GRAFICAR LOS RESULTADOS """
    samp = 80.
    t = np.arange(y.shape[1]) / samp
    
    
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
    ax1.set_title('S')
    ax2.set_title('E_hat')
    ax3.set_title('S_hat')

    f.suptitle('DATA'+ str(i), fontsize=16)
    #subplot1: origina
    for i, wv in enumerate(y):
        ax1.plot( t, wv - 8 * i, "tab:orange",linewidth=2.5)
    plt.tight_layout()
    plt.grid()

    #subplot2: x_hat-> estimación de la entrada (conv kernel con la salida)
    for i, wv in enumerate(x):
        ax2.plot(t,(wv - 8 * i), "tab:red", linewidth=2.5)
        
    plt.tight_layout()
    plt.grid()

    #subplot3: y_hat->
    for i, wv in enumerate(x): #deberia ser y_hat
        ax3.plot(t,wv - 8 * i, c="k",linewidth=2.5)

    plt.tight_layout()
    plt.grid()

    #plt.savefig("figures/multi_cars_impulse.pdf")
    plt.grid()
    plt.show()
    plt.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noncausal_convolution', action='store_false', help='Determines whether causal or noncausal convolution is used.')
    parser.add_argument('--data', default=default_data,type=str,help='Dataset to load.')
    parser.add_argument('--act_function', default = "tanh", type=str, help='Activation function for the last layer of UNet e.g. tanh, relu')
    parser.add_argument('--weights',default = '/checkpoints/best.ckpt', type=str,help='Load weights path.')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='Optimizer for the model e.g. adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 1.0,type=float,help='Percentage dropout to use.')
    parser.add_argument('--deep_win', default = 1024,type=int,help='Number of samples per chunk.')
    parser.add_argument('--integrate', action = 'store_true', help='Indicates if the DAS data and kernel should be integrated.')
    parser.add_argument('--kernel', default = default_kernel, help='Indicates which kernel to use. Receives a <npy> file.')
    parser.add_argument('-pcc','--perform_crosscorrelation', action='store_false', help='Flips kernel in the horizontal axis to perform the cross-correlation. By default perfoms the convolution')
    parser.add_argument('--authors', action='store_true', help='Data from the original work is used, which has a weird shape.')

    opt = parser.parse_args()
    return opt

def main(opt):
	comparison(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

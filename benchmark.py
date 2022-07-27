# Benchmark fista with our model: Resource wise
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
from ISTA import FISTA
import time
import json
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'


def benchargs():
    dsc = 'Scrip to benchmark our network against FISTA resource wise. By default works with our net.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fista', action='store_true', help='Benchmarking for FISTA')
    #parser.add_argument('--image', type=int, default=200)
    parser.add_argument('--trials', type=int, default= 100)
    opt = parser.parse_args()
    return opt

def benchmark(opt):
    noise = 1.0
    deep_win = 1024

    # Load Kernel
    kernel_file = 'kernels/chirp_kernel.npy'
    kernel = np.load(kernel_file)
    kernel /= kernel.max()
    kernel = np.flip(kernel)

    # Load DAS data
    das_file = 'data/CHIRP_DAS_NOFASE_data.h5'
    with h5py.File(das_file,'r') as f:
        _, Nch, Nt = f['strainrate'].shape
        data = f['strainrate'][:, ::].astype(np.float32)

    data /= data.std()
    Nwin = data.shape[1] // deep_win
    # Total number of time samples to be processed
    Nt_deep = Nwin * deep_win


    # Load GroundTruth data 
    gt_file = 'data/deltas_gt.h5'
    with h5py.File(gt_file, 'r') as f:
        gt = f['deltas'][:, ::].astype(np.float32)

    

    if(opt.fista):
        DUT = FISTA(kernel=kernel, lam=0.02)

    else:
        DUT = UNet(
        kernel.astype(np.float32), lam=rho, f0=f0,
        data_shape=(Nch, deep_win, 1), blocks=blocks, AA=False, bn=False, dropout=noise, causal=True, activation_function='tanh'
        )

        DUT.construct()
        DUT.compile()

        # Load weights
        DUT.load_weights('weights/200-epoch-chirp-multi-channel/best.ckpt').expect_partial()

    op_time = []
    error = []
    
    for i in range(opt.trials):
        start_time = time.time()

        if(opt.fista):
            loss, x_hat, y_hat = DUT.solve(data[i], N=7)

        else:
            x_hat, y_hat = DUT.call(data[i][None,:,:])
            x_hat = tf.reshape(x_hat,[24,1024])
            y_hat = tf.reshape(y_hat,[24,1024])

        end_time = time.time()
        op_time.append(end_time-start_time)
        temp_error = x_hat - gt[i]
        error.append(sum([o**2 for o in temp_error])/len(temp_error))
    
        # #DEBUG
        fig, axes = plt.subplots(ncols=1,nrows=4)
        scale = 2

        t = np.arange(x_hat.shape[1]) / samp

        # Plot sample_image
        ax = axes[0]
        ax.set_title("a) Input")
        for j, wv in enumerate(data[i]):
            ax.plot(t, wv - scale * j, c="k")
            break
        
        # Plot x_hat
        ax = axes[2]
        ax.set_title('c) Inferred')
        for j, wv in enumerate(x_hat):
            ax.plot(t, wv - scale * j, c="k")
            break

        ax = axes[3]
        ax.set_title('d) Reconstructed')
        for j, wv in enumerate(y_hat):
            ax.plot(t, wv - scale * j, c="k")
            break

        # Plot y_hat
        ax = axes[1]
        ax.set_title('b) Groundtruth')
        ax.plot(t, gt[i], c='k')
        # for j, wv in enumerate(gt[i]):
        #     ax.plot(t, wv - scale * j, c="k")
        #     break
        fig.tight_layout()
        plt.show()

        break
        


    analysis = dict()
    analysis['time_avg'] = np.average(op_time)
    analysis['time_std'] = np.std(op_time)
    analysis['mse_avg'] = float(np.average(error))
    analysis['mse_std'] = float(np.std(error))


    print("TRIALS INFO:\nAVG TIME:{}\nSTD TIME: {}".format(analysis['time_avg'],analysis['time_std']))
    print("AVG MSE: {}\nSTD MSE {}".format(analysis['mse_avg'], analysis['mse_std']))

    if(opt.fista):
        save_name = 'fista.json'
    
    else:
        save_name = 'ML.json'

    with open(os.path.join('benchmarking',save_name), 'w') as f:
        json.dump(analysis,f)
    
    
        

if __name__ == '__main__':
    opt = benchargs()
    benchmark(opt)
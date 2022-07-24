import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import argparse
# File name

def_path = 'plots/'

def argsOptions():
    dsc = "Generates a plot with the metrics of the training results provided by the JSON file."
    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument("-if", "--InputFile", help="JSON file to process", required=True)
    parser.add_argument("--show", action='store_true', help='Shows the resulting plot')
    parser.add_argument("-op","--OutputPath", default=def_path, help="Output path of the generated plot. By default its {}".format(def_path))
    parser.add_argument('-eps', action='store_true', help='If enabled, saves the figuras as an EPS file. By default it is saved as a png')
    args = parser.parse_args()
    return args

def plotJSON(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    #Check if file exists
    if not os.path.isfile(args.InputFile):
        print("JSON file does not exists!")
        exit(1)

    # Check if output folder exist
    save_folder = os.path.join(ROOT_DIR, def_path)
    if not os.path.isdir(def_path):
        print("Output Path does not exists, creating folder...")
        os.mkdir(save_folder)


    # De-serialize the data
    with open(args.InputFile, 'r') as file:
        history = json.load(file)

    alpha_val = 0.2
    # Filter data

    loss = savgol_filter(history['loss'],101,3)
    val_loss = savgol_filter(history['val_loss'], 101, 3)

    l2 = savgol_filter(history['l2'], 101,3)
    val_l2 = savgol_filter(history['val_l2'],101,3)

    l1 = savgol_filter(history['l1'],101,3)
    val_l1 = savgol_filter(history['val_l1'],101,3)

    fig, axs = plt.subplots(3,1)

    # Plot total loss
    axs[0].plot(history['loss'], c='C0', alpha=alpha_val)
    axs[0].plot(history['val_loss'], c='C1', alpha=alpha_val)
    axs[0].plot(loss, c='C0', label='Train')
    axs[0].plot(val_loss,c='C1', label='Validation')
    axs[0].set_ylabel('Total loss (fidelity + sparsity)')
    axs[0].legend(loc='upper center', ncol=2)

    # Plot Fidelity loss
    axs[1].plot(history['l2'], c='C0', alpha=alpha_val)
    axs[1].plot(history['val_l2'], c='C1', alpha=alpha_val)
    axs[1].plot(l2, c='C0')
    axs[1].plot(val_l2, c='C1')
    axs[1].set_ylabel(r"Fidelity loss ($\ell_2$)")

    # Plot Sparsity val_loss
    axs[2].plot(history['l1'], c='C0', alpha=alpha_val)
    axs[2].plot(history['val_l1'], c='C1', alpha=alpha_val)
    axs[2].plot(l1, c='C0')
    axs[2].plot(val_l1, c='C1')
    axs[2].set_ylabel(r"Sparsity loss ($\ell_1$)")
    axs[2].set_xlabel('Epoch')

    for ax in axs:
        ax.grid()

    # Check format to save
    if args.eps:
        file_format = 'eps'
    else:
        file_format = 'png'

    #Save results
    name = os.path.basename(args.InputFile)
    name = name.replace('json',file_format)
    outfile = os.path.join(save_folder,name)
    plt.savefig(outfile, format = file_format)


    if args.show:
        plt.show()


if __name__ == '__main__':
    args = argsOptions()
    plotJSON(args)

import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def plotArgs():
    dsc = "Plot for the benchmarking outputs"
    parser = argparse.ArgumentParser(description=dsc)
    parser.add_argument('--stats', required=True, help="File with the docker stats")
    parser.add_argument('--gpu', help='File with the GPU information')
    parser.add_argument('--show', action='store_true', help='Show plot results')
    parser.add_argument('--extra', action='store_true', help='Plots avg and std for the data')
    parser.add_argument('--cut', help='Cuts the data at some point')
    parser.add_argument('--csv', action='store_true',help='Outputs CSV files of the readed data')
    args = parser.parse_args()
    return args

def readStats(filename):
    cpu_key = 2
    ram_key = 3
    cpu = []
    ram = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if('CONTAINER' not in line):
                line = line.split()
                cpu_val = float(line[cpu_key].replace('%',''))
                if('GiB' in line[ram_key]):
                    replace_char = 'GiB'
                    ram_val = float(line[ram_key].replace(replace_char,''))
                elif('MiB' in line[ram_key]):
                    replace_char = 'MiB'
                    ram_val = float(line[ram_key].replace(replace_char,''))/1024
                elif('KiB' in line[ram_key]):
                    replace_char = 'KiB'
                    ram_val = float(line[ram_key].replace(replace_char,''))/(1024*1024)
                cpu.append(cpu_val)
                ram.append(ram_val)
    return cpu, ram

def readGPU(filename):
    clk_key = 4
    fb_key = 5
    fb = []
    clk = []
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            if('#' not in line):
                line = line.split()
                clk_val = int(line[clk_key])
                fb_val = int(line[fb_key])
                clk.append(clk_val)
                fb.append(fb_val)
    return clk , fb

def results(args):

    px = 1/plt.rcParams['figure.dpi']
    # Generate figures
    # -> CPU + Memory
    if(args.stats):
        # Stats analysis
        cpu, ram = readStats(args.stats)
        # -> Normalize CPU values in the number of cores
        n_cores = 12
        cpu = [x/n_cores for x in cpu]
        if(args.cut):
            cut = int(args.cut)
            cpu = cpu[0:cut+1]
            ram = ram[0:cut+1]

        cpu_mem, axs = plt.subplots(2,1, figsize=(1920*px, 1080*px))
        # CPU Usage
        axs[0].plot(cpu)
        axs[0].set_xlabel('Tiempo s')
        axs[0].set_ylabel('CPU % Normalizado')
        axs[0].grid()
        axs[0].title.set_text("Utilización de CPU a través del tiempo")

        # Memory
        if(args.extra):
            n_data = len(ram)
            time_axis = np.arange(0, n_data, 1)
            ram_avg = np.average(ram)
            ram_std = np.std(ram)
            y_high = ram_avg+ram_std
            y_low = ram_avg-ram_std
            std_label = 'Desviación Estándar +/- {} Gb'.format(round(ram_std,2))
            avg_label = 'Valor Medio {} Gb'.format(round(ram_avg,2))
            axs[1].fill_between(time_axis,y_high, y_low,color='orange',alpha=0.25, label=std_label)
            axs[1].hlines(ram_avg,0,n_data,'red', label=avg_label)


        axs[1].plot(ram, label='Medición')
        axs[1].set_xlabel('Tiempo s')
        axs[1].set_ylabel('GiB')
        axs[1].grid()
        axs[1].title.set_text("Utilización de memoria RAM a través del tiempo")

        if(args.extra):
            axs[1].legend()

        # Save fig
        fig_name = os.path.join('resulting_plots',os.path.basename(args.stats))
        fig_name = fig_name.replace('.txt', '.png')
        cpu_mem.savefig(fig_name, format='png')

        if(args.csv):
            # Generate column data
            array_save = np.column_stack((cpu,ram))
            file_name = os.path.basename(args.stats).replace('txt','csv')
            np.savetxt(os.path.join('csv',file_name),array_save, delimiter=',', header='CPU%, RAM GiB')

    # -> GPU clock + memory
    if(args.gpu):
        # Gpu analysis
        clk, fb = readGPU(args.gpu)
        gpu, bxs = plt.subplots(2,1, figsize=(1920*px, 1080*px))

        if(args.cut):
            cut = int(args.cut)
            clk = clk[0:cut+1]
            fb = fb[0:cut+1]

        # CLK
        bxs[0].plot(clk)
        bxs[0].set_xlabel('Tiempo s')
        bxs[0].set_ylabel('Frecuencia GPU MHz')
        bxs[0].grid()
        bxs[0].title.set_text('Utilización de procesador GPU a través del tiempo')

        # Memory
        bxs[1].plot(fb)
        bxs[1].set_xlabel('Tiempo s')
        bxs[1].set_ylabel('Memoria Utilizada MB')
        bxs[1].grid()
        bxs[1].title.set_text('Utilización de memoria GPU a través del tiempo')

        # Save fig
        fig_name = os.path.join('resulting_plots',os.path.basename(args.gpu))
        fig_name = fig_name.replace('.csv', '.png')
        gpu.savefig(fig_name, format='png')

        if(args.csv):
            # Generate column data
            array_save = np.column_stack((clk,fb))
            file_name = os.path.basename(args.gpu).replace('txt','csv')
            np.savetxt(os.path.join('csv',file_name),array_save, delimiter=',', header='MHz, MB')


    if(args.show):
        plt.show()

if __name__ == '__main__':
    args = plotArgs()
    results(args)

#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def plot(fn_base):
    file_list = sorted(glob.glob('grid*_' + fn_base + '.ovr'))

    grid_file = np.loadtxt(fn_base + '.grd',
                           usecols=(0,6),
                           dtype=np.dtype([('idx' ,np.int),
                                           ('hden',np.float)]))
    
    datasets = []
    names = []

    for file in file_list:
        datasets.append(
            np.genfromtxt(file, usecols=range(0,8)))
        names.append(file)

    plt.rc('text', usetex=True)
    
    plt.figure()
    for idx, dset in enumerate(datasets):
        plt.xscale('log')
        plt.xlabel('$\log(\mathrm{depth})$')
        plt.ylabel('$x_\mathrm{HII}$')
        # find value of parameter with this index
        param_val = grid_file['hden'][idx]
        plt.plot(dset[:,0], dset[:,7], label='$\log(n_\mathrm{{HI}})={}$'.format(param_val))
        plt.legend()

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename_base',type=str)
    args = parser.parse_args()
    
    plot(args.filename_base)

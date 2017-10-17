#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def plot(fn_base):
    file_list = sorted(glob.glob('grid*_' + fn_base + '.ovr'))

    grid_file = np.loadtxt(fn_base + '.grd', usecols=(0,6), dtype=np.dtype([('idx',np.int),('hden','f')]))
    
    datasets = []
    names = []

    for file in file_list:
        datasets.append(
            np.genfromtxt(file, usecols=range(0,8)))
        names.append(file)

    plt.rc('text', usetex=True)
    
    plt.figure()
    for set, name in zip(datasets,names):
        plt.xscale('log')
        plt.xlabel('$\log(\mathrm{depth})$')
        plt.ylabel('$x_\mathrm{HII}$')
        # get index of file from its name by stripping out all non-digit characters
        number = int(re.sub(r'\D','',name))
        # find value of parameter with this index
        param_val = grid_file[np.where(grid_file['idx'] == number)]['hden'][0]
        plt.plot(set[:,0], set[:,7], label='$\log(n_\mathrm{{HI}})={}$'.format(param_val))
        plt.legend()

    plt.show()

        
    
    
#    record = np.dtype(

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename_base',type=str)
    args = parser.parse_args()
    
    plot(args.filename_base)

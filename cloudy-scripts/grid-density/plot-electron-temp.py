#!/usr/bin/python3

import numpy as np
import matplotlib, matplotlib.pyplot as plt

import cloudy_plots as clplt

if __name__ == '__main__':
    fn_base = 'grid-density'
    cp = clplt.CloudyPlotter(fn_base)

    colourvals = np.linspace(1., 0., cp.nfiles())
    colours = [matplotlib.cm.jet(x) for x in colourvals]

    # Plot of HI fraction vs radius for each H density
    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlabel('$\log(\mathrm{depth/[cm]})$')
    ax.set_ylabel('$x_\mathrm{HI}$')
    ax.set_title('$x_\mathrm{HI}$ as a function of depth')
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterExponent())
    for idx in range(cp.nfiles()):
        # find value of parameter with this index
        hden = cp.get_grid_param(idx, 'hden')
        depth = cp.get_col(idx, 'depth')
        etemp = cp.get_col(idx, 'Te')
        plt.plot(depth, etemp,
                 label='$\log(n_H)={}$'.format(hden),
                 color=colours[idx], marker='.', linestyle='')
        plt.legend()
    plt.show()

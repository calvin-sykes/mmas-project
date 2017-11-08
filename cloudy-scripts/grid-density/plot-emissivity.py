#!/usr/bin/python3

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import cloudy_plots as clplt

if __name__ == '__main__':
    fn_base = 'grid-density'
    cp = clplt.CloudyPlotter(fn_base)

    colourvals = np.linspace(1., 0., cp.nfiles())
    colours = [matplotlib.cm.jet(x) for x in colourvals]

    # Plot of emissivity against radius for each density
    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log(\mathrm{depth/[cm]})$')
    ax.set_ylabel(r'$\epsilon\ (\mathrm{erg\ cm^{-3}\ s^{-1}})$')
    ax.set_title('Emissivity as a function of depth')
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterExponent())
    # Recombination coeffs. as a func of temperature for Halpha
    # from Osterbrock & Ferland, p. 72
    # Case A
    ts = [2500, 5000, 10000, 20000]
    alphas = [6.61e-14 * 3.42, 3.78e-14 * 3.10, 2.04e-14 * 2.86, 1.03e-14 * 2.69]
    alpha_interp = interpolate.interp1d(ts, alphas)
    # Case B
    #alphas = [9.07e-14 * 3.30, 5.37e-14 * 3.05, 3.03e-14 * 2.87, 1.62e-14 * 2.76] 
    #alpha_interp = interpolate.interp1d(ts, alphas)
    # Halpha transition is 1.89eV
    # 1 J = 1e7 erg
    hnu = 3.024e-12 # ergs
    for idx in range(cp.nfiles()):
        alphas = alpha_interp(cp.get_col(idx, 'Te'))
        # emissivity = alpha * np * ne * (h * nu)
        # np = nh * xhii
        npr = cp.get_col(idx, 'hden') * cp.get_col(idx, 'HII')
        ne = cp.get_col(idx, 'eden')
        emiss = alphas * ne * npr * hnu
        #emiss_b = alphas_b * ne * npr * hnu
        # find value of parameter with this index
        hden = cp.get_grid_param(idx, 'hden')
        depth = cp.get_col(idx, 'depth') 
        plt.plot(depth, emiss,
                 label=r'$\log(n_H)={}$'.format(hden),
                 color=colours[idx], marker='.', linestyle='')
        plt.legend()
    plt.show()

#!/usr/bin/python3

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import cloudy_plots as clplt

if __name__ == '__main__':
    fn_base = 'grid-density'
    cp = clplt.CloudyPlotter(fn_base)
    
    # Open the output file to get mean ionisation fractions
    out_file = open(fn_base + '.out')
    lines = out_file.readlines()
    # find lines we care about
    matches = filter(lambda l: "Log10 Mean Ionisation (over radius)" in l, lines)
    # split into words
    matches_exploded = [l.split() for l in matches]
    # extract the strings corresponding to the numerical values
    number_strings = [l[1:4] for l in matches_exploded]
    # Cloudy fails to leave whitespace between numbers sometimes
    # so need to muck around to separate them
    numbers = []
    for ls in number_strings:
        hi = ls[0]
        hii = ''
        if '(H2)' in ls[2]: # the space is missing, so the data looks like '-2.644-11.528'
            hiibeg = ls[1].find('-')
            hiiend = ls[1].rfind('-')
            assert(hiibeg != -1 and hiiend != -1)
            hii = ls[1][hiibeg:hiiend]
        else:
            hii = ls[1]
        numbers.append([10**(float(hi)), 10**(float(hii))]) # values stored as log10
    
    # Ordinary averages aren't very meaningful, because in linear space X_HI approaches 1
    # for the vast majority of the x range --> mean value is near 1
    # So instead I'm taking the average over log(x) space
    myavgs = []
    for idx in range(cp.nfiles()):
        # interpolate over the data
        ionfracs = cp.get_col(idx, 'HI')
        depth = cp.get_col(idx, 'depth')
        func = interpolate.interp1d(depth, ionfracs,
                                    bounds_error=False,
                                    fill_value=(ionfracs[0], ionfracs[-1]))
        # sample over full range [0, 10**30]
        xs = np.logspace(0, 30, 10000)
        ys = func(xs)
        # integrate numerically
        avg = integrate.simps(ys, np.log10(xs)) / 30
        myavgs.append(avg)

    #idx = 0
    #ds = data_ovr[idx]
    #ionfracs = get_col(ds, idx_xhi)
    #func = interpolate.interp1d(get_col(ds, idx_depth),
    #                            ionfracs,
    #                            bounds_error=False,
    #                            fill_value=(ionfracs[0], ionfracs[-1]))
    #xs = np.logspace(0, 30, 10000)
    #ys = func(xs)
    #plt.figure()
    #ax = plt.gca()
    #ax.set_xscale('log')
    #plt.scatter(xs, ys, s=2)
    #plt.plot(get_col(ds, idx_depth), get_col(ds, idx_xhi), color=colours[0], marker='.', linestyle='')

    # Plot of ionisation fraction as a function of H density
    plt.figure()
    ax = plt.gca()
    ax.set_title('$\overline{x}_\mathrm{HI}$ as a function of $n_H$')
    ax.set_xlabel('$\log(n_H)$')
    ax.set_ylabel('$\overline{x}_\mathrm{HI}$')
    hden = [cp.get_grid_param(idx, 'hden') for idx in range(cp.nfiles())]
    plt.plot(hden, myavgs, label='My "log x" averages')
    plt.plot(hden, [n[0] for n in numbers], label='Cloudy averages')
    plt.legend()

    plt.show()

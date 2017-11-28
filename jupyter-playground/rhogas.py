#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
from scipy import integrate, interpolate, optimize
from matplotlib import pyplot as plt, rc, cm, ticker
from sys import version_info
import os
import multiprocessing
import signal
import time

if version_info.major < 3:
    from itertools import repeat, izip
else:
    from itertools import repeat

###############################################################
## Constants used at some point (all converted to cgs units) ##
###############################################################

h = 0.671 # H_0 = 100h km/s/Mpc
solmass = 1.990e30 # kg
solmass *= 1000    # --> g
mu = 0.6 
mp = 1.673e-27 # kg
mp *= 1000     # --> g
G  = 6.674e-11 # m^3/(kg s^2)
G *= 1000      # --> cm^3/(g s^2) 
kB = 1.381e-23 # J/K
kB *= 1e7      # --> erg/K

cm_to_pc = (3e10 * 86400 * 365 * 3.26)
cm_to_kpc = cm_to_pc * 1000

rhocrit = 3 * (100 * h * 1e5 / (cm_to_pc * 1e6))**2 / (8 * np.pi * G)

####################
## T-n_H relation ##
####################

log_nh, log_T = np.loadtxt('T-nH.dat', unpack=True)

log_T = interpolate.UnivariateSpline(log_nh, log_T, s=0.001, k=3)
dlogT_dlognh = log_T.derivative()

def rho(lognh):
    return 10**lognh * mp

def T(lognh):
    return 10**log_T(lognh)

def dT_drho(lognh):
    return (T(lognh) / rho(lognh)) * dlogT_dlognh(lognh)

################
## F function ##
################

# Only consider densities above the mean background density
log_mean_nh = -6.7
log_nh_valid = log_nh[log_nh > log_mean_nh]

def F_integrand(lognh):
    log_int =  T(lognh) / rho(lognh) + dT_drho(lognh)
    # The integral is computed in log(nH) space
    # so a conversion back into rho space is required 
    return log_int * 10**lognh * mp * 2.303 # ln(10)

def Ff(lognh):
    return integrate.quad(F_integrand, log_mean_nh, lognh)[0]

#######################
## Virial quantities ##
#######################

def Rvir(Mvir):
    return ((Mvir * solmass / (200 * rhocrit)) * (3 / (4 * np.pi)))**(1/3)

def Tvir(Mvir):   
    return (mu * mp / (2 * kB)) * (G * Mvir * solmass) / Rvir(Mvir)

################
## G function ##
################

def Gf(rtw, c, Mvir):
    x = rtw * c
    return -2 * Tvir(Mvir) * -np.log(1 + x) / (rtw * (np.log(1 + c) - c / (1 + c)))

##############################
## Concentration parameters ##
##############################

def make_cp_interp():
    cfile = np.loadtxt('Planck_cMz.dat')

    ms = np.log10(10**cfile[:,0] * h * 10**10)
    cps = cfile[:,1]

    return interpolate.interp1d(10**ms, cps)

c = make_cp_interp()

#########################
## rho_gas calculation ##
#########################

class Py2StarHelper:
    def __init__(self, func):
        self.f = func
    def __call__(self, packed_args):
        return self.f(*packed_args)

def kernel(rtw, c, Mvir):
    rhs = Gf(rtw, c, Mvir)
    cond = lambda lognh: np.abs(Ff(lognh) - rhs)   
    #return 10**optimize.brute(cond, ((log_mean_nh, 0),))[0] * mp
    return 10**optimize.minimize_scalar(cond, bounds=(log_mean_nh, 0), method='bounded').x * mp

py2_kernel = Py2StarHelper(kernel)

def rho_gas(rtws, c, Mvir):
    if isinstance(rtws, np.ndarray):
        print('Starting numerical run for M={:.3g}, c={:.3g} with {} workers'.format(Mvir, c, N_CPUS))
        start_time = time.time()
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        worker_pool = multiprocessing.Pool(N_CPUS)
        signal.signal(signal.SIGINT, sigint_handler)
        try:
            if version_info.major < 3:
                rhos = worker_pool.map_async(py2_kernel, izip(rtws, repeat(c), repeat(Mvir)), chunksize=10)
            else:
                rhos = worker_pool.starmap_async(kernel, zip(rtws, repeat(c), repeat(Mvir)), chunksize=10)
            rhos = rhos.get(timeout=600)
        except KeyboardInterrupt:
            print('Aborting...')
            worker_pool.terminate()
            exit()
        else:
            print('Completed in {:.1f}s'.format(time.time() - start_time))
            worker_pool.close()
        worker_pool.join()
    else: # calculate for a single radius
        rhos = kernel(rtws, c, Mvir)
    return rhos

###########################
## Gas mass calculations ##
###########################

def M_gas(rads, rhos):
    if len(rads):
        return 4 * np.pi * integrate.simps(rads**2 * rhos, rads) / solmass
    else:
        return 0

##################
## Main program ##
##################

if __name__ == "__main__":

    fname = 'rhos_gas.dat'
    fname_m = 'rhos_gas_masses.dat'
    found_data = os.path.isfile(fname) and os.path.isfile(fname_m)
    N_CPUS = multiprocessing.cpu_count()

    if not found_data:
        prompt = raw_input('Failed to load data, generate? ')
        if prompt not in set(['yes', 'y', 'Yes']):
            print('Aborting...')
            exit()
        else:
            prompt = raw_input('Number of processes (default = {}) '.format(N_CPUS))
        if prompt is not '':
            try:
                N_CPUS = int(prompt)
            except:
                print('Aborting...')
                exit()
    
    if found_data:
        masses = np.loadtxt('rhos_gas_masses.dat')
        rtws, rhos_gas = np.hsplit(np.loadtxt('rhos_gas.dat'), [1])
        rtws = rtws.flatten()
        rhos_gas = rhos_gas.T
    else:
        #############
        ## EDIT ME ##
        #############
        masses = np.logspace(8, 9.65, 20) # in M_sun
        rtws = np.geomspace(1e-3, 30, 1000)

    T200s = Tvir(masses)
    R200s = Rvir(masses)
    concentrations = c(masses)

    ##################
    ## Calculations ##
    ##################
    
    if not found_data:
        print('Calculating for {} radius values in range ({}, {})'.format(len(rtws), rtws.min(), rtws.max()))
        all_start_time = time.time()
        rhos_gas = np.array([rho_gas(rtws, cp, M200) for M200, cp
                                 in zip(masses, concentrations)])
        print(rhos_gas)
        print('Completed all calculations in {:.1f}s'.format(time.time() - all_start_time))
        np.savetxt('rhos_gas.dat', np.column_stack((rtws, rhos_gas.T)))
        np.savetxt('rhos_gas_masses.dat', masses)        

    # virial radii
    rhos_r200 = np.array([rho_gas(1, c, Mvir) for c, Mvir in zip(concentrations, masses)])

    # radii where Mgas == Mvir * fbar
    fbar = 0.167 # universal baryon fraction
    mass_targets = fbar * masses

    Rbars = np.empty_like(masses)
    rhos_Rbar = np.empty_like(masses)

    for idx, mt in enumerate(mass_targets):
        loc = np.argmin([np.abs(mt - M_gas(rtws[:trial_loc] * R200s[idx], rhos_gas[idx][:trial_loc])) for trial_loc in range(len(rtws))])
        Rbars[idx] = rtws[loc] * R200s[idx]
        rhos_Rbar[idx] = rhos_gas[idx][loc]

    ###########
    ## Plots ##
    ###########
    
    if True:
        rtws_within_r200 = rtws[rtws <= 1]
        gas_masses = [M_gas(rtws_within_r200 * rv, rg[rtws < 1]) for rg, rv in zip(rhos_gas, R200s)]
        
        colourvals = np.linspace(0., 1., len(rhos_gas))
        colours = [cm.rainbow(x) for x in colourvals]
        sm = cm.ScalarMappable(cmap=cm.rainbow)
        sm.set_array(np.log10(gas_masses))

        fig, (axm, axcb) = plt.subplots(1, 2, figsize=(13,12), gridspec_kw = {'width_ratios':[12, 1]})
    axm.set_xlim(-0.5, 2.5)
    axm.set_ylim(-7, -0.5)
    axm.set_xlabel(r'$\log_{10}(r/\mathrm{kpc})$')
    axm.set_ylabel(r'$\log_{10}(n_H/\mathrm{cm^{-3}})$')
    for idx, rg in enumerate(rhos_gas):
        kpc_rads = rtws * R200s[idx] / cm_to_kpc
        axm.plot(np.log10(kpc_rads), np.log10(rg / mp), color=colours[idx])    
    # plot virial radii
    axm.plot(np.log10(R200s / cm_to_kpc), np.log10(rhos_r200 / mp), '--', c='k')
    axm.text(1.15, -5.75, r'$r_{200}$', rotation=50)
    # plot fbar * M200 radii
    axm.plot(np.log10(Rbars / cm_to_kpc), np.log10(rhos_Rbar / mp), '--', c='k')
    axm.text(2.0, -6.2, r'$r_\mathrm{bar}$', rotation=60)
    # plot mean nH
    axm.axhline(log_mean_nh, c='k', ls='--')
    axm.text(-.25, -6.6, r'$\bar{n}_H$')
    # plot colourbar
    axcb.yaxis.tick_right()
    axcb.yaxis.set_label_position('right')
    axcb.set_ylabel(r'$\log_{10}(M_\mathrm{gas}/M_\odot)$')
    axcb.xaxis.set_major_locator(ticker.NullLocator())
    axcb.xaxis.set_major_formatter(ticker.NullFormatter())
    for idx, mg in enumerate(gas_masses):
        axcb.axhline(np.log10(mg), c=colours[idx])
    plt.show()

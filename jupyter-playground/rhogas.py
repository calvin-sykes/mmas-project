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
import argparse

if version_info.major < 3:
    from itertools import repeat, izip
else:
    from itertools import repeat

###############################################################
## Constants used at some point (all converted to cgs units) ##
###############################################################

h = 0.673 # H_0 = 100h km/s/Mpc
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

def rho(lognh):
    """
    Convert log10(number density) to physical density in g cm^-3
    """
    return 10**lognh * mp

class DensityTempRelation:
    """
    Encapsulates a density-temperature relation obtained from a text file containing a list of paired log(n_H), log(T) values.
    Loads the data and constructs an interpolation T(n_H) and its derivative dT/dn_H
    """

    def __init__(self, model_str):
        """
        Construct a density-temperature relation.
        
        model_str: String containing the name of the data file to load. The path searched is 'data/nH_T_' + model_str + '.dat'
        """
        try:
            log_nH, log_T = np.loadtxt('data/nH_T_{:s}.dat'.format(model_str.lower()), unpack=True)
        except IOError:
            print('{:s} density-temperature relation data not found\nAborting...'.format(model_str))
            exit(1)
        else:
            print('Loading {:s} density-temperature parameter relation data'.format(model_str.title()))
            self.__logTfunc = interpolate.UnivariateSpline(log_nH, log_T, s=0.001, k=3)
            self.__dlogTfunc = self.__logTfunc.derivative()

    def T(self, lognh):
        """
        Return the temperature for a given log10(number density)
        """
        return 10**self.__logTfunc(lognh)

    def dT_drho(self, lognh):
        """
        Return the derivative of temperature wrt number density for a given log10(number density)
        """
        return (self.T(lognh) / rho(lognh)) * self.__dlogTfunc(lognh)

# Global variable to hold the density-temperature relation object.
# Defined now so that it is visible to functions which need to use it.
T_rel = None

################
## F function ##
################

# Only consider densities above the mean background density
log_mean_nh = -6.7

def F_integrand(lognh):
    """
    Compute the value of T/rho + dT/drho for a given log10(number density).
    """
    log_int =  T_rel.T(lognh) / rho(lognh) + T_rel.dT_drho(lognh)
    # The integral is computed in log(nH) space
    # so a conversion back into rho space is required
    # 2.303 = ln(10)
    return log_int * 10**lognh * mp * 2.303

def Ff(lognh):
    """
    Numerically integrate the F_integrand function, with lower limit given by the mean background density.

    lognh: The upper limit of integration.
    """
    return integrate.quad(F_integrand, log_mean_nh, lognh)[0]

#######################
## Virial quantities ##
#######################

def Rvir(Mvir):
    """
    Calculate the virial radius for a given virial mass.

    Mvir: The virial mass in units of solar masses.

    Returns: The virial radius in centimetres.
    """
    return ((Mvir * solmass / (200 * rhocrit)) * (3 / (4 * np.pi)))**(1/3)

def Tvir(Mvir):
    """
    Calculate the virial temperature for a given virial mass.

    Mvir: The virial mass in units of solar masses.

    Returns: The virial temperature in Kelvin.
    """
    return (mu * mp / (2 * kB)) * (G * Mvir * solmass) / Rvir(Mvir)

################
## G function ##
################

def Gf(rtw, c, Mvir):
    """
    Calculate G(r)=-2*Tvir*a(r), where a(r) is the acceleration profile.

     rtw: The dimensionless radial coordinate r/Rvir.
       c: The concentration parameter of the halo.
    Mvir: The virial mass of the halo.
    """
    x = rtw * c
    return -2 * Tvir(Mvir) * -np.log(1 + x) / (rtw * (np.log(1 + c) - c / (1 + c)))

##############################
## Concentration parameters ##
##############################

class MassConcentrationRelation:
    """
    Encapsulates a virial mass-concentration parameter relation obtained from a text file containing a list of paired Mvir, c values.
    Loads the data and constructs an interpolation c(Mvir)
    """

    def __init__(self, model_str):
        """
        Construct a mass-concentration relation.
        
        model_str: String containing the name of the data file to load. The path searched is 'data/Mvir_c_' + model_str + '.dat'
        """
        try:
            ms, cs = np.loadtxt('data/Mvir_c_{:s}.dat'.format(model_str.title()), unpack=True)
        except IOError:
            print('{:s} mass-concentration parameter relation data not found\nAborting...'.format(model_str))
            exit(1)
        else:
            print('Loading {:s} mass-concentration parameter relation data'.format(model_str.title()))
            self.__interp = interpolate.interp1d(ms, cs)

    def cparam(self, Mvir):
        """
        Return the concentration parameter for a given virial mass.
        """
        return self.__interp(Mvir)

#########################
## rho_gas calculation ##
#########################

class Py2StarHelper:
    """
    Utility class to unpack a tuple of arguments.
    Used to emulate the `starmap` function from Python 3 if the code is run with Python 2.
    """
    
    def __init__(self, func):
        self.f = func
    def __call__(self, packed_args):
        return self.f(*packed_args)

def kernel(rtw, c, Mvir):
    """
    Calculate the density rho which solves to the expression F(rho) - G(r).
    
     rtw: The dimensionless radial coordinate r/Rvir.
       c: The concentration parameter of the halo.
    Mvir: The virial mass of the halo.

    Returns: The density which solves the expression above for the given parameters
    """
    rhs = Gf(rtw, c, Mvir)
    cond = lambda lognh: np.abs(Ff(lognh) - rhs)   
    #return 10**optimize.brute(cond, ((log_mean_nh, 0),))[0] * mp
    return 10**optimize.minimize_scalar(cond, bounds=(log_mean_nh, 0), method='bounded').x * mp

if version_info.major < 3:
    py2_kernel = Py2StarHelper(kernel)

def rho_gas(rtws, c, Mvir):
    """
    Calculate the gas density profile for a particular DM halo.

    rtws: The dimensionless radial coordinates r/Rvir at which to calculate densities. May be a numpy array or a single value.
       c: The concentration parameter of the halo.
    Mvir: The virial mass of the halo.

    Returns: A numpy array of densities if an array of radii were provided; a single density otherwise.
    """
    if isinstance(rtws, np.ndarray):
        print('Starting calculations for M={:.3g}, c={:.3g} with {} workers'.format(Mvir, c, N_CPUS))
        start_time = time.time()
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN) # This stops Ctrl-C breaking while multiple processes are running
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
            exit(0)
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
    """
    Calculate the total gas mass given a radial density profile.

    rads: Array of radius coordinates in physical units.
    rhos: Array of density values, at each radius in rads, in physical units.

    Returns: Total gas mass in solar masses.
    """
    if len(rads):
        return 4 * np.pi * integrate.simps(rads**2 * rhos, rads) / solmass
    else:
        return 0

##################
## Main program ##
##################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c' , '--mass-conc-rel', dest='conc', required='true', type=str, help='The mass-concentration relation to use. Either "eagle" or "prada"')
    parser.add_argument('-t' , '--nH-temp-rel', dest='nHT', required='true', type=str, help='The hydrogen density-temperature relation to use. Either "relhic" or "sphcloudy"')
    parser.add_argument('-f', '--force-recalc', dest='force', action='store_true', help='Regenerate density profiles even if data is already found')
    parser.add_argument('-p', '--plot', dest = 'plot', action='store_true', help='Produce a plot of the gas density profiles when calculations are complete')
    args = parser.parse_args()

    fname_ext = '.dat'
    fname_base = 'output/rhos_gas'+ '_{:s}_{:s}'.format(args.conc.lower(), args.nHT.lower())
    fname_rho = fname_base + fname_ext
    fname_M = fname_base + '_masses' + fname_ext
    fname_Rv = fname_base + '_Rvirs' + fname_ext
    found_data = os.path.isfile(fname_rho) ^ (args.force)

    N_CPUS = multiprocessing.cpu_count()

    if not found_data:
        prompt = raw_input('Failed to load data, generate? ')
        if prompt not in set(['yes', 'y', 'Yes']):
            print('Aborting...')
            exit(0)
        else:
            prompt = raw_input('Number of processes (default = {}) '.format(N_CPUS))
        if prompt is not '':
            try:
                N_CPUS = int(prompt)
            except:
                print('Aborting...')
                exit(1)
    
    if found_data:
        masses = np.loadtxt(fname_M)
        nhaloes = len(masses)
        rtws, rhos_gas, _ = np.hsplit(np.loadtxt(fname_rho), [1, 1 + nhaloes]) # ignore temperatures for now
        R200s = np.loadtxt(fname_Rv).flatten()
        rtws = rtws.flatten()
        rhos_gas = rhos_gas.T
    else:
        #############
        ## EDIT ME ##
        #############
        masses = np.logspace(8, 9.65, 21) # in M_sun
        rtws = np.geomspace(1e-3, 30, 1000)
        R200s = Rvir(masses)

    c_rel = MassConcentrationRelation(args.conc)
    concentrations = c_rel.cparam(masses)

    T_rel = DensityTempRelation(args.nHT)

    ##################
    ## Calculations ##
    ##################
    
    if not found_data:
        print('Calculating for {} radius values in range ({}, {})'.format(len(rtws), rtws.min(), rtws.max()))
        all_start_time = time.time()
        rhos_gas = np.array([rho_gas(rtws, cp, M200) for M200, cp
                                 in zip(masses, concentrations)])
        print('Completed all calculations in {:.1f}s'.format(time.time() - all_start_time))
        np.savetxt(fname_rho, np.column_stack((rtws, rhos_gas.T, T_rel.T(np.log10(rhos_gas / mp)).T)))
        np.savetxt(fname_M, masses)
        np.savetxt(fname_Rv, R200s)
    elif not args.plot:
        print('Nothing to do, exiting.')
        exit(0)

    # densities at virial radii
    rhos_R200 = np.array([rho_gas(1, c, Mvir) for c, Mvir in zip(concentrations, masses)])

    # densities at radii where Mgas == Mvir * fbar
    fbar = 0.167 # universal baryon fraction omega_b / omega_M
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
    
    if args.plot:
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
        axm.plot(np.log10(R200s / cm_to_kpc), np.log10(rhos_R200 / mp), '--', c='k')
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

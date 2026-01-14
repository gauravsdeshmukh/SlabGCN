"""Calculate surface coverage based on a stat. mech. approach."""

import numpy as np

from .constants import h, N_A, kB, c, e

# Define standard functions
def cm1_to_Hz(wavenum, c=c):
    """Convert wavenumber in cm^-1 to Hz."""
    wavenum *= 100 # Convert to m^-1
    freq = c / (1 / wavenum)
    return freq

def joule_to_eV(joules):
    """Convert joules to eV."""
    return joules / e

def ZPE(nu, h=h):
    """Calculate ZPE."""
    E_ZPE = h * np.sum(np.array(nu)) / 2
    return E_ZPE

def U_trans(T, kB=kB):
    """Calculate translational internal energy."""
    return 3 * kB * T / 2

def U_rot(T, lin, kB=kB):
    """Calculate rotational internal energy."""
    if lin == "linear":
        return kB * T
    else:
        return 3 * kB * T / 2
    
def U_vib(T, nu, kB=kB):
    """Calculate vibrational internal energy."""
    beta = 1 / (kB * T)
    nu = np.array(nu)
    _U_vib = np.sum(h * nu / (np.exp(beta * h * nu) - 1))
    return _U_vib

def S_trans(T, P, m, h=h, kB=kB):
    """Calculate translational entropy."""
    h_bar = h / (2 * np.pi)
    beta = 1 / (kB * T)
    _S_trans = kB * np.log((m / (2 * np.pi * beta * h_bar**2))**(3 / 2) * (1 / (beta * P))) +\
                5 * kB / 2
    return _S_trans

def S_rot(T, I, sigma, lin, h=h, kB=kB):
    """Calculate rotational entropy."""
    beta = 1 / (kB * T)
    h_bar = h / (2 * np.pi)
    if lin == "linear":
        _S_rot = kB * np.log(I / (sigma * beta * h_bar**2)) + kB
    else:
        _S_rot = kB * np.log((np.sqrt(np.pi) / sigma) * \
                             (2 / (h_bar**2 * beta))**(3/2) * I**(1/2)) + 3 * kB / 2
                             
    return _S_rot
                             
def S_vib(T, nu, h=h, kB=kB):
    """Calculate vibrational entropy."""
    beta = 1 / (kB * T)
    _S_vib = np.sum(- h * nu / (2 * T) - kB * np.log(1 - np.exp(-h * nu * beta))) + \
                (1 / T) * np.sum((1 / 2) * h * nu + (h * nu / (np.exp(beta * h * nu) - 1)))
    return _S_vib

# Chemical potentials of H, H2S and adsorbed S
def mu0_H(T, kB=kB, N_A=N_A):
    """Chemical potential correction for H in gas-phase."""
    # Frequency
    nu_H =  cm1_to_Hz(4299.931172)
    
    # Other constants
    I_H = 4.71e-48 # kg m^2
    sigma_H = 1
    m_H = 2.01568e-3 / N_A #kg / atom
    
    # ZPE
    _E_ZPE = ZPE(nu_H)
    
    # Internal energy
    _U_trans = U_trans(T)
    _U_rot = U_rot(T, "linear")
    _U_vib = U_vib(T, nu_H)
    
    # Entropy
    _S_trans = S_trans(T, 1.013e5, m_H)
    _S_rot = S_rot(T, I_H, sigma_H, "linear")
    _S_vib = S_vib(T, nu_H)
    _S = _S_trans + _S_rot + _S_vib
    
    # Enthalpy
    _H = _U_trans + _U_rot + (_U_vib + _E_ZPE) + kB * T
    
    # Chemical potential
    _mu = _H - T * (_S)
    _mu = joule_to_eV(_mu)
    
    return _mu

def mu0_H2S(T, kB=kB, N_A=N_A):
    """Chemical potential correction for H2S in gas-phase."""
    # Frequencies
    nu_H2S =  np.array([
                cm1_to_Hz(2658.061382), cm1_to_Hz(2642.844905), cm1_to_Hz(1168.828092)
                ])
    
    # Other constants
    I_H2S = 5.380424e-140 # kg m^2
    sigma_H2S = 1
    m_H2S = 34.076e-3 / N_A #kg / atom
    
    # ZPE
    _E_ZPE = ZPE(nu_H2S)
    
    # Internal energy
    _U_trans = U_trans(T)
    _U_rot = U_rot(T, "nonlinear")
    _U_vib = U_vib(T, nu_H2S)
    
    # Entropy
    _S_trans = S_trans(T, 1.013e5, m_H2S)
    _S_rot = S_rot(T, I_H2S, sigma_H2S, "nonlinear")
    _S_vib = S_vib(T, nu_H2S)
    _S = _S_trans + _S_rot + _S_vib
    
    # Enthalpy
    _H = _U_trans + _U_rot + (_U_vib + _E_ZPE) + kB * T
    
    # Chemical potential
    _mu = _H - T * (_S)
    _mu = joule_to_eV(_mu)
    
    return _mu

def mu_ads(T, nu=358.13, kB=kB, N_A=N_A):
    """Chemical potential correction for bound adsorbate."""
    # ZPE
    nu = cm1_to_Hz(nu)
    _E_ZPE = ZPE(nu)
    
    # Vib U
    _U_vib = U_vib(T, nu)
    
    # Vib S
    _S_vib = S_vib(T, nu)
    
    # mu
    _mu = joule_to_eV(_E_ZPE + (_U_vib) - (T * _S_vib))
    return _mu

def mu_gas(T, P_ratio, kB=kB):
    """Chemical potential of ideal gas (correction)."""
    _mu_gas = kB * T * np.log(P_ratio)
    _mu_gas_eV = joule_to_eV(_mu_gas)

    return _mu_gas_eV

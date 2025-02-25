# import necessary packages
from parameters_general import *


#semiconductor parameters GaAs
m_e = 0.0665 * m0
m_h = 0.1106 * m0
M = m_e+m_h
m_eff = m_e*m_h/M
E_G = 1.514
epsilon_s_stat = 12.46 #static dielectric constant
epsilon_s_opt = 10.58 #optical dielectric constant
n_ref = np.sqrt(epsilon_s_opt) #optical density of material

#parameters Quantum well
E_1s = -0.008895642372159323 #1s exciton binding energy
d_cv = 0.87 #effective dipole moment
d = 7.6 * 1.3 #effective sample thickness
gamma = 0.5*10 **(-3)/hbar #inverse phonon scattering dephasing time




omega_0 = E_G/hbar
omega_1 = omega_0 +E_1s/hbar #- 0.012903883497007542 #Trägerfrequenz E1
omega_2 = omega_1 #Trägerfrequenz E2


tdim = 200000
tmax = 20000
dt = tmax/tdim
tlist = np.linspace(dt, tmax, tdim)


#m = 3  #ordnung der raeumlichen fourier trafo


energyresolution = 1000
E_start = -0.015 -E_G #+ 0.012903883497007542 * hbar
E_stop = 0.005 -E_G #+ 0.012903883497007542 * hbar

omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

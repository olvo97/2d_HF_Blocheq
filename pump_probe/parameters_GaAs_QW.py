# import necessary packages
import numpy as np
from parameters_general import *


#GaAs parameters
m_e = 0.0665 * m0
m_h = 0.1106 * m0
M = m_e+m_h
mu = m_e*m_h/M
E_G = 1.475
epsilon_s_stat = 12.46 #static dielectric constant
epsilon_s_opt = 10.58 #optical dielectric constant
n_medium = np.sqrt(epsilon_s_opt) #optical density of material
n_surrounding = 1 #optical density of surrounding medium (air)

#MQW and laser parameters
w_p = 300000/np.sqrt(2*np.log(2)) #spot size radius pump pulse in nm
w_t = 200000/np.sqrt(2*np.log(2)) #spot size radius probe pulse in nm
gamma_rep = 5000 * 10**(-15) #repetition rate in 1/fs
spacing = 28.6 #spacing between quantum wells in nm
spacing_0 = 4.6 #spacing between initial air_GaAs interface and first quantum well in nm
z_2 = 500320 #total thickness of GaAs sample in nm

#fitted parameters
d_cv = 0.76  #effective dipole moment
gamma_lattice = 0.9*10 **(-3)/hbar
d = 7.6 * 1.3 #effective sample thickness


#electric field and RWA central frequency
exciton_energy = 0.008833833833833835
omega_pump = E_G/hbar - exciton_energy/hbar + 0.0005/hbar#Trägerfrequenz E1, leicht detuned von 1s peak #Ist abgespaltene Frequenz in RWA!
omega_probe = omega_pump
detuning = 0.0 #additional detuning
FWHM = 1349 #temporal full width half maximum of pulse intensity
sigma_pump =FWHM/(2*np.sqrt(np.log(2)))#temporal standard deviation of pump pulse
sigma_probe = 90 #temporal standard deviation of probe pulse
E_probe = 0.00002 *2*np.pi*hbar/d_cv #Amplitude Probe field
t_pump = 5000 #temporal center of pump pulse


#numerical grid, k-grid is variable and defined in main file
phidim = 2000
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim 

tdim = 200000
tmax = 20000
dt = tmax/tdim
tlist = np.linspace(dt, tmax, tdim)

energyresolution = 1000
E_start = -0.015 + 0.012903883497007542 * hbar
E_stop = 0.005 + 0.012903883497007542 * hbar
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

#intraband relaxation parameters
Q_eqrange = 200 #number of wavevectors until which the fermi distribution is evaluated for determining chem_pot
T = 6 #initial temperature in Kelvin
gamma_inter =  1/100#/hbar #100 fs dephasing time
chem_pot = E_G-0.007 #initial chemical potential MUST BE BELOW ENERGY GAP!!!


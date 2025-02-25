# import necessary packages
import numpy as np
from parameters_general import *

#GaAs parameters
m_e = 0.067 * m0
m_h = 0.1106 * m0
M = m_e+m_h
mu = m_e*m_h/M
E_G = 1.514
epsilon_s_stat = 12.7 #static dielectric constant
epsilon_s_opt = 12.96 #optical dielectric constant
n_medium = np.sqrt(epsilon_s_opt) #optical density of material
n_surrounding = 1 #optical density of surrounding medium (air)

#fitted parameters
d_cv = 0.575  #effective dipole moment
gamma_lattice = 1/526
d = 8 #effective sample thickness


#electric field and RWA central frequency
omega_0 = E_G/hbar - 0.013877187614636867#Trägerfrequenz E1, resonant zu 1s peak #Ist abgespaltene Frequenz in RWA!
FWHM = 770 #temporal full width half maximum of pulse intensity
sigma = FWHM/(2*np.sqrt(np.log(2)))#temporal standard deviation of electric field gauss pulse
t_c = 3000 #temporal center of electric field gauss pulse


#numerical grid, k-grid is variable and defined in main file
phidim = 2000
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim 

tdim = 100000
tmax = 10000
dt = tmax/tdim
tlist = np.linspace(dt, tmax, tdim)

energyresolution = 1000
E_start = -0.015 + 0.012903883497007542 * hbar
E_stop = 0.005 + 0.012903883497007542 * hbar
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

#MQW parameters (have no purpose, if n_wells =1 and reflections = False)
z_2 = 500320 #total thickness of GaAs sample in nm
spacing = 28.6 #spacing between quantum wells
spacing_0 = 4.6 #spacing between initial air-material interface and first quantum well
    
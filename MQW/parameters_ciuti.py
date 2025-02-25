# import necessary packages
import numpy as np
from parameters_general import *

#GaAs parameters
m_e = 0.067 * m0
m_h = 0.1675 * m0
M = m_e+m_h
mu = m_e*m_h/M
E_G = 1.514
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
gamma_lattice = 0.0008947554337612708
d = 15 #effective sample thickness


#electric field and RWA central frequency 
omega_0 = E_G/hbar - 0.00841341341341341/hbar#Trägerfrequenz E1, resonant zu 1s peak #Ist abgespaltene Frequenz in RWA!
sigma = 90#367.6926103285312 #FWHM/(2*np.sqrt(np.log(2)))#temporal standard deviation of electric field gauss pulse
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
E_start = -0.025 + 0.012903883497007542 * hbar
E_stop = 0.005 + 0.012903883497007542 * hbar
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

# import necessary packages
import numpy as np
from parameters_general import *

#RADIATIVES DAMPING IN FUNCTIONS AUSSCHALTEN!!!
#GaAs parameters
m_e = 0.0665 * m0
m_h = 0.1106 * m0
M = m_e+m_h
mu = m_e*m_h/M
E_G = 1.514
epsilon_s_stat = 12.46 #static dielectric constant
#epsilon_s_opt = 10.58 #optical dielectric constant
n_medium = 1 #optical density of material
#n_surrounding = 1 #optical density of surrounding medium (air)

#fitted parameters
d_cv = 0.76  #effective dipole moment
gamma_lattice = 1*10 **(-3)/hbar
d = 8 * 1.3 #effective sample thickness


#electric field and RWA central frequency 
omega_0 = E_G/hbar - 8.77/1000/hbar #Trägerfrequenz E1, leicht detuned von 1s peak #Ist abgespaltene Frequenz in RWA!
FWHM = 2000 #temporal full width half maximum of pulse intensity
sigma =FWHM/(2*np.sqrt(np.log(2)))#temporal standard deviation of electric field gauss pulse
t_c = 3000 #temporal center of electric field gauss pulse

#numerical grid, k-grid is variable and defined in main file
phidim = 4000
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim 

tdim = 80000
tmax = 8000
dt = tmax/tdim
tlist = np.linspace(dt, tmax, tdim)

energyresolution = 1000
E_start = -0.015
E_stop = 0.005
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

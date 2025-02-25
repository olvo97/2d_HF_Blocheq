# import necessary packages
import numpy as np


#semiconductor parameters
hbar = 0.658212196
m0 = 5.6856800 
m_e = 0.0665 * m0
m_h = 0.1106 * m0
M = m_e+m_h
mu = m_e*m_h/M
e = 1
A = 1
k_B = 8.61745*10**(-5)
        
E_G = 1.514
E_1s = -0.008895642372159323

epsilon_0 = 5.526308*10**(-2) 
epsilon_s_stat = 12.46 #static dielectric constant
epsilon_s_opt = 10.58 #optical dielectric constant
d_cv = 0.87 #effective dipole moment
d = 7.6 * 1.3 #effective sample thickness
gamma = 0.5*10 **(-3)/hbar
c = 2.997925*10**2 # vacuum light speed
n_ref = np.sqrt(epsilon_s_opt) #optical density of material


omega_0 = E_G/hbar
omega_1 = omega_0 +E_1s/hbar #- 0.012903883497007542 #Trägerfrequenz E1
omega_2 = omega_1 #Trägerfrequenz E2

# test parameters
Qdim = 1000
phidim = 2000
Qmax = 1.5

dQ = Qmax/Qdim
qlist  = np.linspace(Qmax/Qdim, Qmax, Qdim)
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim

E_1 = 2*np.pi*hbar/d_cv 
sigma1 =750#width of gaussian envelope for pump field
t1 = 5000 #time delay of pump field

E_2 = 0.0001 *2*np.pi*hbar/d_cv #Amplitude Probe field
sigma2 = 90#width of gaussian envelope for probe field
t2 = 4000 #time delay of probe field


tdim = 200000
tmax = 20000
dt = tmax/tdim
tlist = np.linspace(dt, tmax, tdim)


#m = 3  #ordnung der raeumlichen fourier trafo


energyresolution = 1000
E_start = -0.015 -E_1s #+ 0.012903883497007542 * hbar
E_stop = 0.005 -E_1s #+ 0.012903883497007542 * hbar

#E_start = -0.025
#E_stop = 0.05
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

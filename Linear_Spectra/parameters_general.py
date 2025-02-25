import numpy as np 

#general parameters in semiconductor units
hbar = 0.658212196
m0 = 5.6856800 #resting electron mass
e = 1
k_B = 8.61745*10**(-5)
epsilon_0 = 5.526308*10**(-2) 
c = 2.997925*10**2 # vacuum light speed
bohr_radius = 4*np.pi*epsilon_0*hbar**2/(e**2 * m0)
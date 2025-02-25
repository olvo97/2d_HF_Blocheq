import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pickle
import sys
from functions import *
from parameters_GaAs_QW import *



# test parameters
Qdim = 500
phidim = 2000
Qmax = 1.5
d = float(sys.argv[1])

dQ = Qmax/Qdim
qlist  = np.linspace(Qmax/Qdim, Qmax, Qdim)
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim

#adapt omega_list to old version of RWA around E_G
energyresolution = 1000
E_start = -0.015 
E_stop = 0.005
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

EWs, EVs = solve_(wannier_matrix(qlist, philist, m_eff, epsilon_s_stat, d), qlist) 

#E_R_0 = E_R_mu(EWs,EVs,0)
#abs_1s = get_absorption(E_R_0, energyresolution)
#plt.plot(1000* np.linspace(E_start,E_stop,energyresolution),abs_1s, label = '1s exciton')

#E_R_1 = E_R_mu(EWs,EVs,1)
#abs_2s = get_absorption(E_R_1, energyresolution)
#plt.plot(1000*np.linspace(E_start,E_stop,energyresolution),abs_2s, label = '2s exciton')
#plt.plot(hbar*np.array(omega_list), np.array([theta(i) for i in omega_list]), label = 'analytical_free')

#format plot
#plt.xlabel(r'$E-Eg$ [meV]')
#plt.ylabel('absorption')
#plt.legend()
#plt.savefig('results/absorption_linear_ex_Qmax={}_Qdim={}.pdf'.format(Qmax, Qdim))
#plt.show()

#save data..

with open(r"results/wellwidths/Qmax={}_Qdim={}_d={}_exciton_EWs.pickle".format(Qmax, Qdim,d), 'wb') as output_file:
    pickle.dump(EWs, output_file)
with open(r"results/wellwidths/Qmax={}_Qdim={}_d={}_exciton_EVs.pickle".format(Qmax, Qdim,d), 'wb') as output_file:
    pickle.dump(EVs, output_file)
#with open(r"results/wellwidths/Qmax={}_Qdim={}_d={}_exciton_absorption_1s.pickle".format(Qmax, Qdim,d), 'wb') as output_file:
#    pickle.dump(abs_1s, output_file)
#with open(r"results/wellwidths/Qmax={}_Qdim={}_d={}_exciton_absorption_2s.pickle".format(Qmax, Qdim,d), 'wb') as output_file:
 #   pickle.dump(abs_2s, output_file)
#with open(r"results/Qmax={}_Qdim={}_eh_spectrum_coulomb.pickle".format(Qmax, Qdim), 'wb') as output_file:
  #  pickle.dump(alpha_eh_coulomb, output_file)
#with open(r"results/Qmax={}_Qdim={}_eh_spectrum_free.pickle".format(Qmax, Qdim), 'wb') as output_file:
 #   pickle.dump(alpha_eh_free, output_file)

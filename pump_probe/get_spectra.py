from functions import *
from parameters_GaAs_QW import *
import pickle
import matplotlib.pyplot as plt

power = 30
delays = np.array([-3000, -2500, -2000, -1500, -1333, -1166, -1000, -833, -666, -500, -333, -166, 0, 166, 333, 500, 666, 833, 1000, 1500, 2000, 2500, 3000, 3500, 4000],dtype = float)
#delays = [float(i) for i in range(-3000, 4500, 500)]

energyresolution = 2000
E_start = -0.015 + 0.012903883497007542 * hbar
E_stop = 0.005 + 0.012903883497007542 * hbar
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)
absorptions = []

#True_gamma_inter=0.01
for dt in delays:
    #load file
    with open(r"results/general/dt={}_E_pump={}_sqrt_power_conversion_factor=0.15_d_cv=0.87_gamma=0.5_Qdim=700_Qmax=1.5_quadratic_grid_intra_rel=False_Ps.pickle".format(dt,power), 'rb') as input_file:
        P_ts = pickle.load(input_file)
    #compute absorptions
    absorption_LB = get_absorption_probe(P_ts[2], t_pump + dt, tlist,omega_list, E_probe, sigma_probe, omega_probe, n_medium, pulse_shape = 'gauss', method = 'lambert_beer')
    absorption_RT = get_absorption_probe(P_ts[2], t_pump + dt, tlist,omega_list, E_probe, sigma_probe, omega_probe, n_medium, pulse_shape = 'gauss', method = '1-R-T')
    #save absorptions
    with open(r"results/general/dt={}_E_pump={}_sqrt_power_conversion_factor=0.15_d_cv=0.87_gamma=0.5_Qdim=700_Qmax=1.5_quadratic_grid_intra_rel=False_absorption_LB.pickle".format(dt,power), 'wb') as output_file:
        pickle.dump(absorption_LB, output_file)
    with open(r"results/general/dt={}_E_pump={}_sqrt_power_conversion_factor=0.15_d_cv=0.87_gamma=0.5_Qdim=700_Qmax=1.5_quadratic_grid_intra_rel=False_absorption_1-R-T.pickle".format(dt,power), 'wb') as output_file:
        pickle.dump(absorption_RT, output_file)
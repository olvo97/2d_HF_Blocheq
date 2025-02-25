from functions import *
from runge_kutta import solve_runge_kutta
import matplotlib.pyplot as plt
import pickle
import sys
import time
import datetime

start = time.time() #measure time for execution of individual steps

parameter_set = str(sys.argv[1]) #'schülzgen', 'ciuti', 'eigen', 'henry'
if parameter_set == 'schuelzgen':
    from parameters_schuelzgen import *
elif parameter_set == 'eigen':
    from parameters_eigen import *
elif parameter_set == 'ciuti':
    from parameters_ciuti import *
elif parameter_set == 'henry':
    from parameters_henry import *

'''
parameter sets 'schülzgen' and 'ciuti' imply their own power conversions to reproduce the respective papers
'schülzgen: give fluence in uJ/cm² as argument
'eigen': give power in uW as argument
for parameter set 'eigen', linear and true (sqrt) fluence to E-field conversion can be applied. Additionally, a fluence correction factor can be defined for true conversion

'''  
intervalls_total = 10 #total number of separate simulations following each other for time series. More intervalls require less RAM
return_n_distr = True  #specify, if the final carrier distribution should be saved
#define parameters from input
Qdim = int(sys.argv[2])
Qmax = float(sys.argv[3])
quadratic_scaling =  bool(int(sys.argv[5])) 
reflections = bool(int(sys.argv[8]))

linear_conversion = bool(int(sys.argv[10]))
pf_corr = float(sys.argv[9])
conv_string = ''
if parameter_set == 'schuelzgen':
    initial_reflection = 0.68
    if reflections:
        ref_par = 0 #don't count the first reflection twice
    pulse_strength = np.sqrt(initial_reflection*float(sys.argv[4])*pf_corr*6.24157*1e-2*4*np.sqrt(np.pi)*sigma/(epsilon_0*c*n_medium))
    conv_string = '_pf_corr={}'.format(pf_corr)

elif parameter_set == 'eigen':
    if linear_conversion:
        pulse_strength = 2*np.pi*hbar/d_cv*float(sys.argv[4])/10*0.037
        conv_string = '_linear_E_conversion_0.037/10'
    else:
        if reflections:
            substract_initial_reflection = False #don't count the first reflection twice
        else:
            substract_initial_reflection = True
        pulse_strength = E0_from_power(float(sys.argv[4]), pf_corr, sigma, gamma_rep, w_t, w_p, n_medium, substract_initial_reflection) 
        conv_string = '_true_E_conversion_pf_corr={}'.format(pf_corr)
        

elif parameter_set == 'ciuti':
    pulse_strength = np.pi*hbar/(2*d_cv)*float(sys.argv[4])
    conv_string = '_V_screened_d=15_final'

elif parameter_set == 'henry':
    pulse_strength = np.pi*hbar/(d_cv)*float(sys.argv[4])
    conv_string = ''


n_wells = int(sys.argv[6])
z_2 = float(sys.argv[7])


#define fixed and derived parameters  
pulse_shape = 'gauss'
qlist  = np.linspace(Qmax/Qdim, Qmax, Qdim)
qdrat_str = '_linear_grid'
if quadratic_scaling:
    qlist = qlist**2/Qmax
    qdrat_str = '_quadratic_grid'
V_kq = V_matrix(qlist, philist, epsilon_s_stat, d)

print(r"Filenames: results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_...".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str))
print(str(datetime.timedelta(seconds=time.time()-start)),': Initialized Parameters and Coulomb matrix')

#define wrapper for function
if not reflections and n_wells == 1: #use more efficient singlepulse implementation
    print('Use_SQW_implementation')
    @jit(nopython = True)
    def ddt_nonlinear_singlepulse_wrapper(t,y):
        return ddt_singlepulse_SQW(t, y, V_kq, qlist, pulse_strength, sigma, t_c, d_cv, omega_0, E_G, mu, gamma_lattice, n_medium, pulse_shape)
else:
    print('Use_MQW_implementation')
    @jit(nopython = True)
    def ddt_nonlinear_singlepulse_wrapper(t,y):
        return ddt_singlepulse_MQW(t,y, n_wells, d_cv, V_kq, omega_0, E_G,mu, gamma_lattice, t_c ,pulse_strength, sigma, qlist, 
                            n_medium, n_surrounding, z_2,spacing, spacing_0, reflections, pulse_shape)
#solve ODE
P_ts = [[]*n_wells]
N_ts = [[]*n_wells]
for intervall in range(intervalls_total):
    if intervall == 0:
        y0 = np.zeros(2*n_wells*Qdim, dtype = 'complex128')
    else:
        y0 = Y[-1,:]
    tlist_ = partition_list(tlist,intervalls_total)[intervall]
    Y = solve_runge_kutta(ddt_nonlinear_singlepulse_wrapper, tlist_, y0)
    print('Solved Integration_intervall{}'.format(intervall))
    #compute macroscopic polarization for each quantum well
    p_t_ = np.transpose(Y[:,:n_wells*Qdim])
    n_t_ = np.transpose(Y[:,n_wells*Qdim:])
    [P_ts[i].extend(P(p_t_[Qdim*i:Qdim*(i+1)], d_cv = d_cv, qlist=qlist)) for i in range(n_wells)]
    [N_ts[i].extend(N(n_t_[Qdim*i:Qdim*(i+1)], qlist=qlist)) for i in range(n_wells)]
    if return_n_distr:
        ns_t2 = n_t_[:Qdim].transpose()[-1]
        with open(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_final_n_distr.pickle".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str), 'wb') as output_file:
            pickle.dump(ns_t2, output_file)

print(str(datetime.timedelta(seconds=time.time()-start)),': Computed macroscopic polarizations and occupations')

#write macroscopic polarizations and occupations
with open(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_polarizations.pickle".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str), 'wb') as output_file:
    pickle.dump(P_ts, output_file)
with open(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_occupations.pickle".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str), 'wb') as output_file:
    pickle.dump(N_ts, output_file)
print('Saved macroscopic polarizations and occupations')
if pulse_shape == 'gauss':
    plt.fill(tlist,gauss_pulse(tlist,pulse_strength, sigma, t_c), 'lightgray')
if pulse_shape == 'sigma':
    plt.fill(tlist,sigma_pulse(tlist,pulse_strength, sigma, t_c), 'lightgray')
for i,pol in enumerate(P_ts):
    plt.plot(tlist, np.abs(pol), label = 'i={}'.format(i))
plt.xlabel('time [fs]')
plt.ylabel(r'$|\tilde{P}_i|$')
plt.legend()
#plt.savefig(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_polarizations.pdf".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str))
plt.clf()

if pulse_shape == 'gauss':
    plt.fill(tlist,gauss_pulse(tlist,pulse_strength, sigma, t_c), 'lightgray')
if pulse_shape == 'sigma':
    plt.fill(tlist,sigma_pulse(tlist,pulse_strength, sigma, t_c), 'lightgray')
for i,N in enumerate(N_ts):
    plt.plot(tlist, np.real(N)*(bohr_radius**2), label = 'i={}'.format(i))
plt.xlabel('time [fs]')
plt.ylabel(r'$\tilde{N}_i\; [a_0^2]$')
plt.legend()
#plt.savefig(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_occupations.pdf".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str))
plt.clf()
print('Plotted macroscopic polarizations and occupations')
if sigma < 150:
    if not reflections and n_wells == 1:
        abs = get_absorption_SQW(P_ts[0], t_c, tlist,omega_list, pulse_strength, sigma, omega_0, n_medium, pulse_shape, method = 'lambert_beer')
    else:
        abs = get_absorption_MQW(P_ts, t_c, tlist, omega_list, pulse_strength, sigma, n_surrounding, n_medium, z_2, spacing, spacing_0, omega_0, reflections, pulse_shape, method = 'lambert_beer')
    print(str(datetime.timedelta(seconds=time.time()-start)),': Computed absorption spectrum')

    with open(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_absorption.pickle".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str), 'wb') as output_file:
        pickle.dump(abs, output_file)

    plt.plot(1000* hbar*(omega_list), abs, label = '{}QWs'.format(n_wells))
    plt.plot(1000* hbar*(omega_list), np.abs(fouriertrafo(gauss_pulse(tlist,pulse_strength, sigma, t_c), omega_list, tlist)), label = 'E-field')

    plt.xlabel(r'$E-E_G$ [meV]')
    plt.ylabel('absorption')
    plt.legend()
    plt.savefig(r"results/{}/Qdim={}_Qmax={}_power{}{}_n_wells{}_z2{}_reflections{}{}_absorption.pdf".format(parameter_set,Qdim, Qmax, sys.argv[4],conv_string, n_wells,z_2,reflections,qdrat_str))
    print('Saved absorption Spectrum')

print(str(datetime.timedelta(seconds=time.time()-start)),': Done')
from parameters_GaAs_QW import *
from functions import *
import sys
import pickle
import time
from runge_kutta import solve_runge_kutta
import matplotlib.pyplot as plt

mode = str(sys.argv[9])#'splitting_study' #determine, which kind of experiment should be performed
plotting = False #direkt ergebnisse plotten
#-----------------------------------------------------------------------------------------------------------------------------------
if mode == 'splitting_study':
    pump_detuning = -0.0014/hbar
else: 
    pump_detuning = 0

start = time.process_time()
#Arguments:
if mode == 'gain':
    delta_t = 4000
    d_cv = float(sys.argv[1])
    gamma_lattice = float(sys.argv[5])/hbar
    quadratic_scaling = True
    comment = '_d_cv={}_gamma_lattice={}_no_induced_em'.format(d_cv,gamma_lattice)
else:
    delta_t = float(sys.argv[1])
    quadratic_scaling =  bool(int(sys.argv[5])) 
    comment = ''
conversion_linear = bool(int(sys.argv[7]))
conv_factor = float(sys.argv[8])
if conversion_linear:
    E_pump = 2*np.pi*hbar/d_cv*0.038/10*float(sys.argv[2]) 
    conv_str = '_linear_power_conversion_factor=0.0038'
elif mode == 'splitting_study':
    power = float(sys.argv[2]) * 15/4.244
    E_pump = E0_from_power(power, conv_factor , sigma_pump, gamma_rep, w_t, w_p, n_medium, substract_initial_reflection = True)
    conv_str = '_sqrt_power_conversion_factor={}'.format(conv_factor)
else:
    E_pump = E0_from_power(float(sys.argv[2]), conv_factor , sigma_pump, gamma_rep, w_t, w_p, n_medium, substract_initial_reflection = True)
    #d_cv = 0.87
   # gamma_lattice = 0.0005/hbar
    conv_str = '_sqrt_power_conversion_factor={}'.format(conv_factor)
Qdim = int(sys.argv[3])
Qmax = float(sys.argv[4])

    
phidim = 2000
qlist = np.linspace(Qmax/Qdim, Qmax, Qdim)

qdrat_str = 'linear_grid'
if quadratic_scaling:
    qlist = qlist**2/Qmax
    qdrat_str = '_quadratic_grid'

intraband_relaxation =  bool(int(sys.argv[6])) 
intra_rel_str = '_intra_rel=False'
if intraband_relaxation: 
    intra_rel_str = '_intra_rel=True_gamma_inter={}'.format(gamma_inter)

t_probe = t_pump + delta_t
Q_eqrange = Qdim

philist= np.arange(0,2*np.pi,2*np.pi/phidim)

V_kq = V_matrix(qlist, philist, epsilon_s_stat, d)
# #test the E-field
# plt.plot(hbar*omega_list*100,fouriertrafo(E1(tlist,E_1)), label = 'pump_field')
# plt.plot(hbar*omega_list*100,fouriertrafo(E2(tlist,E_2, delta_t)), label = 'probe_field')
# plt.xlabel('E-E_G [meV]')
# plt.ylabel('Electric Field Strength')
# plt.show()

n0_list = np.zeros(Qdim)#equilibrium_occupation(qlist, chem_pot) #initialize ground state fermi distribution 

#solve ODE
@jit(nopython = True)
def ddt_pump_probe_wrapper(t,y, chem_pot, T, n0_list):
    return ddt_pump_probe(t,y, chem_pot, T, n0_list, V_kq, E_pump, t_pump, omega_pump, sigma_pump, E_probe, t_probe, omega_probe, sigma_probe, qlist, d_cv, Q_eqrange, mu, E_G, gamma_lattice, n_medium, intraband_rel = intraband_relaxation, gamma_inter = gamma_inter, pump_detuning = pump_detuning)
tspan = np.array([0,tmax])
p0 = np.zeros(5*Qdim, dtype = 'complex128')
Y_nonlinear, potlist, Tlist, n0_list = solve_runge_kutta(ddt_pump_probe_wrapper, tlist, p0, chem_pot, T, n0_list)

#retrieve results from state vector
indices_p = [[2,-1],[1,0],[0,1]]
indices_n = [[-1,1],[0,0]]
t = tlist
p_ts = [np.transpose(Y_nonlinear[:, i*Qdim:(i+1)*Qdim]) for i in range(3)]
n_ts = [np.transpose(Y_nonlinear[:, i*Qdim:(i+1)*Qdim]) for i in range(3, 5)]
P_ts = np.array([P(p_t_, d_cv, qlist) for p_t_ in p_ts])
N_ts = np.array([N(n_t_, qlist) for n_t_ in n_ts])

ns_t2 = n_ts[1].transpose()[int((t_pump+delta_t)*tdim/tmax)]

print('Time taken for numerical integration: ' ,time.process_time()-start, ' s')
start = time.process_time()

with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_Ns.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
    pickle.dump(N_ts, output_file)

with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_Ps.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
    pickle.dump(P_ts, output_file)
    
with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_nsattprobe.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
    pickle.dump(ns_t2, output_file)

with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_TandChempot.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
    pickle.dump((Tlist,potlist), output_file)


absorption_probe = get_absorption_probe(P_ts[2], t_probe, tlist,omega_list, E_probe, sigma_probe, omega_probe, n_medium, pulse_shape = 'gauss', method = 'lambert_beer')

with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_absorption_probe.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
    pickle.dump(absorption_probe, output_file)

if plotting:
    plt.plot(qlist,ns_t2)
    plt.savefig(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_ndistrattprobe.pdf".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str))
    plt.clf()

    plt.plot(tlist,potlist)
    plt.savefig(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_chemical_potential.pdf".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str))
    plt.clf()

    plt.plot(tlist,Tlist)
    plt.savefig(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_temperature.pdf".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str))
    plt.clf()

    #Plot desired results
    for n in [2,1]: 
        index = indices_p[n]
        #print(index[0])
    # plt.plot(tlist[:2000],np.imag(P_ts[n])[:2000], label = 'Im[P({},{})]'.format(index[0],index[1]))
    # plt.plot(tlist[:2000],np.real(P_ts[n])[:2000], label = 'Re[P]({},{})]'.format(index[0],index[1]))
        plt.plot(t,np.abs(P_ts[n]), label = 'abs[P({},{})]'.format(index[0],index[1]))
    for n in [1]:
        index = indices_n[n]
        plt.plot(t,np.real(N_ts[n]), label = 'N({},{})'.format(index[0],index[1]))


    plt.plot(t, 0.1*np.abs(gauss_pulse(t, E_0 = E_pump, sigma = sigma_pump, t_center = t_pump,detuning =  pump_detuning)),'--',label = 'E_pump(t)')
    plt.plot(t, 0.1*np.abs(gauss_pulse(t, E_0 = E_probe, sigma = sigma_probe, t_center = t_probe)),'--',label = 'E_probe(t)')
    plt.xlabel('t[fs]')
    plt.ylabel('a.u.')
    plt.legend()
    #plt.show()
    plt.savefig(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_time_evolution.pdf".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str))
    plt.clf()
    #compute spectra
    absorption_probe = get_absorption_probe(P_ts[2], t_probe, tlist,omega_list, E_probe, sigma_probe, omega_probe, n_medium, pulse_shape = 'gauss', method = 'lambert_beer')

    with open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_absorption_probe.pickle".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'wb') as output_file:
        pickle.dump(absorption_probe, output_file)

    plt.plot(1000 * hbar*(omega_list- 0.012903883497007542), absorption_probe, label = 'alpha_probe')
    #plt.plot(100 * hbar*omega_list, 0.1*fouriertrafo(E1(t,pump_strength)), label = 'pump_field')

    plt.xlabel('E-E_G [meV]')
    plt.ylabel('alpha')
    plt.legend()
    #plt.show()
    plt.savefig(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_absorption_probe.pdf".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str))
    print('Time taken for spectrum calculation and plotting: ',time.process_time()-start, ' s')

    file = open(r"results/{}/{}dt={}_E_pump={}{}_Qdim={}_Qmax={}{}{}_parameters.txt".format(mode,comment,delta_t, sys.argv[2], conv_str, Qdim, Qmax, qdrat_str, intra_rel_str), 'w') 
    file.write('Q_eqrange ={}, Qmax = {}, gamma_inter = {}, E_2 = E_probe = {}; sigma_probe = {}, E_1 = E_pump = {}, sigma_pump = {}, tdim = {}, tmax = {}, energyresolution = {}, omega_pump = {}, omega_probe = {}'.format(Q_eqrange, Qmax, gamma_inter, E_probe, sigma_probe, E_pump, sigma_pump, tdim, tmax, energyresolution, omega_pump, omega_probe)) 
    file.close() 
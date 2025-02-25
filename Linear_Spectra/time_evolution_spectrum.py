import sys
sys.path.append('/home/olivervoigt/Python/Eigen/Core_functions')
from parameters import *
from functions import *
from runge_kutta import solve_runge_kutta
import matplotlib.pyplot as plt
import pickle
 
Qdim = int(sys.argv[1])
Qmax = float(sys.argv[2])

pulse_shape = 'gauss'
tdim = 100000
tmax = 10000
delta_t = tmax/tdim
qlist  = np.linspace(Qmax/Qdim, Qmax, Qdim)
tlist = np.linspace(delta_t, tmax, tdim)
V_kq = V_matrix(qlist)

@jit(nopython = True)
def ddt_nonlinear_singledir_wrapper(t,y):
    return ddt_nonlinear_singledir(t,y,V_kq, qlist = qlist, E_2=E_2, sigma2 = sigma2, t2 = t1, shape = pulse_shape)
#solve ODE
p0 = np.zeros(2*Qdim, dtype = 'complex128')
Y = solve_runge_kutta(ddt_nonlinear_singledir_wrapper, tlist, p0)

p_t_ = np.transpose(Y[:,:Qdim])
n_t_ = np.transpose(Y[:,Qdim:])
P_t = P(p_t_, qlist, tdim)
N_t = N(n_t_, qlist, tdim)


#ft = fouriertrafo(P_t, omega_list, tlist)
abs = get_spectrum(P_t, tlist, omega_list = omega_list, t2=t1, shape = pulse_shape)

with open(r"results/Qmax={}_Qdim={}_tmax={}_time_spectrum_coulomb.pickle".format(Qmax, Qdim, tmax), 'wb') as output_file:
    pickle.dump(abs, output_file)

plt.plot(1000* hbar*(omega_list- 0.012903883497007542), abs)
plt.xlabel(r'$E-E_G$ [meV]')
plt.ylabel('absorption')
plt.legend()
plt.savefig('results/absorption_linear_time_Qmax={}_Qdim={}.pdf'.format(Qmax, Qdim))
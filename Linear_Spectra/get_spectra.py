import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pickle
import sys
#sys.path.append('/home/olivervoigt/Python/Eigen/Core_functions')
from parameters import *



# test parameters
Qdim = int(sys.argv[1])
phidim = 2000
Qmax = float(sys.argv[2])
d = float(sys.argv[3])
d_cv = float(sys.argv[4])

dQ = Qmax/Qdim
qlist  = np.linspace(Qmax/Qdim, Qmax, Qdim)
philist = np.arange(0,2*np.pi,2*np.pi/phidim)
dphi = 2*np.pi / phidim

#adapt omega_list to old version of RWA around E_G
energyresolution = 1000
E_start = -0.015 
E_stop = 0.005
omega_list = np.linspace(E_start/hbar,E_stop/hbar,energyresolution)

#Implementation in electron-hole-picture
@jit(nopython = True)
def intervall(mode, Maximum, dim):
    if mode == "radial":
        L = np.linspace(Maximum/dim, Maximum, dim) # eigentlich mit endpoint = True, aber geht nicht in nopython mode
        #L = np.arange(0, Maximum, Maximum/dim)
    if mode == "phi":
        L = np.linspace(0,2*np.pi,dim)
    return L

# #define matrix and field
# @jit(nopython=True)
# def V(q):
#     '''2d 1/q Coulomb potential'''        
#     return e**2/(2*epsilon_0*epsilon_s_stat*q)

@jit(nopython = True)
def V(q):
    qq = np.absolute(q)
    epsilon_s_bar = 1
    brack = epsilon_s_bar * qq * d
    return (e**2/(A*2*epsilon_0 * epsilon_s_stat * qq) * 1/(brack*(4*np.pi**2+brack**2))
            * (8*np.pi**2 + 3*(brack)**2 - (32*np.pi**4 * (1- np.exp(-brack)))/(brack * (4*np.pi**2 + brack**2))))

@jit(nopython=True)
def matrix_linear_full(omega,coulomb = True):
        '''generates matrix that describes Linear optical response of QW'''
        matrix = np.zeros((Qdim,Qdim), dtype='complex_')
        for n in range(Qdim):
            for j in range(Qdim):
                if n == j:
                    matrix[n][j] += hbar *(omega- hbar * (0.5 / mu) * (qlist[n]**2) + 1j*gamma)# diagonalbeiträge
                # every matrix element has this contribution
                phi_integral = 0   
                if coulomb: 
                    for i in range(phidim):
                        # round is added to avoid negative values
                        q_diff = np.sqrt(np.round(qlist[n]**2+qlist[j]**2-2*qlist[n]*qlist[j]*np.cos(philist[i]),10))
                        
                        if q_diff >= dQ:                    
                            phi_integral += V(q_diff)# + 1j* d_cv**2 * (omega_0)/(2* epsilon_0*c*n_ref) #JE NACHDEM WO ICH ES DAZU PACKE ÄNDERT SIE DIE ABS AMPLITUDE (HIER WIE EXCITON UNTEN WIE TIME DOMAIN)             
                phi_integral *= dQ*qlist[j]*dphi*A
                phi_integral /= (2*np.pi)**2
                matrix[n][j] += phi_integral 
                matrix[n][j] += 1j* d_cv**2 * (omega_0)/(2* epsilon_0*c*n_ref) * dQ * qlist[j] *A/(2*np.pi)# radiative damping
        return matrix


@jit(nopython = True)
def E_R(coulomb = True):
    #compute inverted system matrices for each omega
    M = []
    for omega in omega_list:
        matrix = matrix_linear_full(omega,coulomb)            
        M.append(np.linalg.inv(matrix))
        #compute weighted sums of inverted matrix rows for each omega
    E_R = []
    for m in M:
        sum = 0
        for i, vec in enumerate(m):
            sum -=np.sum(vec)*d_cv**2 * qlist[i] * dQ * A/(2*np.pi) #summe über k, dcv*P
        sum *= 1/(2*epsilon_0 * c*n_ref) * 1j * omega_0 #E_R(P) faktoren
        E_R.append(sum)
    return np.array(E_R)

def get_absorption(E_R, energyresolution):
    R = np.absolute(E_R)**2
    T = np.absolute(np.ones(energyresolution, dtype = 'complex_')+E_R) **2
    return np.ones(energyresolution)- R - T
    
def theta(x):
    if x< 0:
        return 0
    else:
        return (x+E_G/hbar) *  d_cv**2 *mu/(2*c*n_ref*epsilon_0*hbar**2) 

#implementation in exciton-picture
@jit(nopython = True)
def wannier_matrix(): 
    '''generates matrix that describes eigenvalue problem of Wannier equation'''
    matrix = np.zeros((Qdim,Qdim), dtype='complex_')
    for n in range(Qdim):
        for j in range(Qdim):
            if n == j:
                matrix[n][j] += hbar * hbar * 0.5 / mu * qlist[n]**2
            # every matrix element has this contribution
            phi_integral = 0    
            for i in range(phidim):
                # round is added to avoid negative values
                q_absolute = np.sqrt(np.round(qlist[n]**2+qlist[j]**2-2*qlist[n]*qlist[j]*np.cos(philist[i]),10))
                if q_absolute > dQ:
                    phi_integral += V(q_absolute)#/ (eps0 *(1 + 140*q_absolute))#epsilon(q_absolute, TMDC)
                    # phi_integral += self.potential(q_absolute)            
            phi_integral *= dQ*qlist[j]*dphi
            phi_integral /= (2*np.pi)**2
            matrix[n][j] -= phi_integral    
    return matrix


#@jit(nopython = True)
def solve(matrix):
    '''diagonalizes matrix and return eigenvalues and eigenvectors'''
    EWs, EVs = np.linalg.eig(matrix)
    #print("Es wurden {EW} Eigenwerte und {EV} Eigenvektoren gefunden.".format(EW=len(EWs), EV=len(EVs)))
    negativcounter = 0
    for EW in EWs:
        if EW.real < 0:
            negativcounter += 1
            #print(EW)
    #print("davon haben {neg} negativen Realteil".format(neg=negativcounter))
    return EWs, EVs

@jit(nopython = True)
def normalize(EVs):
    ''' normalizes the eigensystem'''
    for i in range(len(EVs)):
        area = 0.
        for j in range(len(EVs)):
            q = qlist[j] 
            area += q*dQ/2/np.pi*np.conjugate(EVs[j,i])*EVs[j,i]
        for j in range(len(EVs)):
            if area != 0:
                EVs[j,i]/= np.sqrt(area)
    return EVs

#@jit(nopython = True)
def sort(EWs,EVs):
    '''sort eigenvectors after eigenvalue''' 
    #EWs, EVs = (list(t) for t in zip(*sorted(zip(EWs, EVs))))
    for i in range(Qdim):
        for j in range(i,Qdim):
            if (EWs[j]<EWs[i]):
                EWs[[i,j]] = EWs[[j,i]]
                EVs[:,[i, j]] = EVs[:,[j, i]]
    return EWs, EVs

#@jit(nopython = True)
def solve_(matrix):
    '''Wrapper for solve, normalize and sort'''
    EWs, EVs = solve(matrix)
    EVs = normalize(EVs)
    EWs, EVs = sort(EWs,EVs)
    return EWs, EVs

#@jit(nopython = True)
def phi0(EVs,mu):
    ''' returns the exciton orbital mu value at r = 0'''
    phi0 = 0.
    for j in range(Qdim):
        q = qlist[j] 
        phi0 += q*dQ/2/np.pi*EVs[j,mu]    
    return phi0    

def chi(EWs,EVs, mu, omega):
    return -d_cv**2/epsilon_0 * np.abs(phi0(EVs,mu))**2*(1/(hbar*(omega-EWs[mu]/hbar + 1j*gamma + 1j* d_cv**2 * np.abs(phi0(EVs,mu))**2 *omega_0/(2*hbar*c*n_ref*epsilon_0))))#+1/(hbar*(-omega-E_G/hbar-EWs[mu]/hbar - 1j*gamma - 1j*d_cv**2 * np.abs(phi0(EVs,mu))**2 *omega/(epsilon_0*c*n_ref))))

        
def E_R_mu(EWs,EVs, mu):
    '''E_R/E_0, emitted field per incoming field'''
    E_R = []
    for omega in omega_list:    
        E_R.append(1j*omega_0/(2*epsilon_0 * c*n_ref)* epsilon_0 * chi(EWs,EVs,mu,omega))
    return E_R

def E_R_full(EWs,EVs):
    E_R = np.zeros(energyresolution,dtype = 'complex128')
    for mu in range(len(EWs)):
        E_R = E_R + np.array(E_R_mu(EWs,EVs, mu))
    return E_R


def alpha_1s(omega, EWs, EVs, full = 'False'):
    gamma_R = np.abs(phi0(EVs,0))**2
    alpha = (omega+E_G/hbar)/(c*n_ref) * (gamma + gamma_R)/((hbar*omega-EWs[0])**2+hbar*(gamma+gamma_R)**2)
    return alpha

#compute reflection and transmission with coulomb
#alpha_eh_coulomb = get_absorption(E_R(coulomb = True), energyresolution)
#plt.plot(np.linspace(E_start,E_stop,energyresolution),alpha_eh_coulomb, label = 'Matrix inversion coulomb')


#compute reflection and transmission without coulomb
#alpha_eh_free = get_absorption(E_R(coulomb = False), energyresolution)
#plt.plot(np.linspace(E_start,E_stop,energyresolution),alpha_eh_free, label = 'Matrix inversion free')

EWs, EVs = solve_(wannier_matrix()) 
E_R_0 = E_R_mu(EWs,EVs,0)
#plt.plot(np.linspace(E_start,E_stop,energyresolution),get_absorption(E_R_0, energyresolution), label = '1s exciton')
#plt.plot(hbar*np.array(omega_list), np.array([theta(i) for i in omega_list]), label = 'analytical_free')

#format plot
#plt.xlabel('E-Eg [eV]')
#plt.ylabel('absorption')
#plt.legend()
#plt.savefig('results/absorption_linear_1s_Qmax={}_Qdim={}.pdf'.format(Qmax, Qdim))

#save data

#with open(r"results/Qmax={}_Qdim={}_exciton_EWs.pickle".format(Qmax, Qdim), 'wb') as output_file:
 #   pickle.dump(EWs, output_file)
#with open(r"results/Qmax={}_Qdim={}_exciton_EVs.pickle".format(Qmax, Qdim), 'wb') as output_file:
#    pickle.dump(EVs, output_file)

#with open(r"results/dipolemoments/Qmax={}_Qdim={}_d={}_dcv={}_eh_spectrum_coulomb.pickle".format(Qmax, Qdim, d, d_cv), 'wb') as output_file:
 #   pickle.dump(alpha_eh_coulomb, output_file)
#with open(r"results/Qmax={}_Qdim={}_eh_spectrum_free.pickle".format(Qmax, Qdim), 'wb') as output_file:
    #pickle.dump(alpha_eh_free, output_file)

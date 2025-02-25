from parameters_general import *
import numpy as np
from numba import jit


##############################################################################################################################################################    
#incident electric fields

@jit(nopython = True)
def gauss_pulse(t, E_0, sigma, t_center, detuning = 0, chirp = 0):
     ''' Generates gaussian envelope function 
     t: time of list of times
     E_0: electric field Amplitude E_0
     sigma: temporal standard deviation
     t_center: temporal center of Gauss pulse
     detuning (optional): detuning from central frequency of rotating wave approximation
     chirp (optional): linear frequency chirp

     returns value of gaussian field envelope at time t, or list of values if list is passed as t argument
     '''
     return E_0/(np.sqrt(2*np.pi)*sigma)*np.exp(-((t-t_center)**2/(2*sigma**2))) * np.exp(-1j*(detuning/hbar)*(t-t_center)) *np.exp(-1j*(chirp)*(t-t_center)**2) 


@jit(nopython = True)
def sigma_pulse(t, E_0, sigma, t_center, detuning = 0, chirp = 0):
    ''' Generates sigma envelope function 
     t: time of list of times
     E_0: electric field Amplitude E_0
     sigma: temporal standard deviation
     t_center: temporal center of sigma pulse
     detuning (optional): detuning from central frequency of rotating wave approximation
     chirp (optional): linear frequency chirp

     returns value of gaussian field envelope at time t, or list of values if list is passed as t argument
     '''
    return E_0/(np.pi*sigma)*1/np.cosh((t-t_center)/sigma) * np.exp(-1j*(detuning/hbar)*(t-t_center)) *np.exp(-1j*(chirp)*(t-t_center)**2) 

def E0_from_power(power, pf_corr, sigma, gamma_rep, w_t, w_p, n_ref, substract_initial_reflection = True):
    '''computes the gaussian field amplitude corresponding to a given laser power, according to paper appendix
    power (float): laser power in uW
    pf_corr (float): correction factor to the pump fluence, must be adapted to compensate for deviation from experiment
    sigma (float): temporal standard deviation of gaussian electric field in fs
    gamma_rep (float): laser repetition rate in 1/fs
    w_t (float): spot size radius probe pulse in nm 
    w_p (float): spot size radius pump pulse in nm (because rabi oszillations in experiment are measured by differential absorption after pump)
    n_ref (float): background refractive index of medium
    substract_initial_reflection (bool): If True, first reflection at air/GaAs interface is substracted. MUST BE SET TO FALSE, IF REFLECTIONS = TRUE IN DDT_SINGLEPULSE_MQW!!! 

    returns: gaussian field amplitude E_0
    '''
    power *= 6.24151*10**(-3) #umrechnung uW in eV/fs
    A_probe = np.pi * w_t ** 2
    fluence = 1/(A_probe * gamma_rep) * (1-np.exp(-2* w_t**2/w_p**2))/(1-np.exp(-2)) * pf_corr * power
    r = (n_ref-1)/(n_ref+1)
    if not substract_initial_reflection:
        r = 0
    return (1-r) * np.sqrt((4*np.sqrt(np.pi)*sigma)/(epsilon_0*c*n_ref) * fluence)

##############################################################################################################################################################    
#Coulomb Matrix

@jit(nopython = True)
def V_basic(q, epsilon_s_stat): 
    '''2d 1/q Coulomb potential
    q: distance in spacial frequency space
    epsilon_s_stat: static dielectric constant of background material
    '''        
    return e**2/(2*epsilon_0*epsilon_s_stat*np.absolute(q))

@jit(nopython = True)
def V(q, epsilon_s_stat, d):
    ''' n=1 QW Coulomb potential for equal dielectric constant at both sides of QW
    q: distance in spacial frequency space
    epsilon_s_stat: static dielectric constant of background material
    d: effective quantum well thickness
    '''  
    qq = np.absolute(q)
    epsilon_s_bar = 1
    brack = epsilon_s_bar * qq * d
    return (e**2/(2*epsilon_0 * epsilon_s_stat * qq) * 1/(brack*(4*np.pi**2+brack**2))
            * (8*np.pi**2 + 3*(brack)**2 - (32*np.pi**4 * (1- np.exp(-brack)))/(brack * (4*np.pi**2 + brack**2))))
               #+ 64*np.pi**4*np.sinh(brack/2)/(brack*(4*np.pi**2+brack**2))*(gamma_minus + delta_plus - 2*delta_minus * np.exp(-brack))/(delta_minus*np.exp(-brack) - gamma_plus * np.exp(brack))))

@jit(nopython = True)    
def V_matrix(qlist, philist, epsilon_s_stat, d):
    '''
    qlist: list of absolute values for 2d-impulse wavevectors
    philist: list of angles for 2d-impulse wavevectors
    epsilon_s_stat: static dielectric constant of background material
    d: effective quantum well thickness

    returns: Coulomb matrix V_kq
    '''
    Qdim = len(qlist)
    phidim = len(philist)
    dphi = philist[1]-philist[0] # assumes equidistant phi-grid
    V_matrix = np.zeros((Qdim,Qdim), dtype = 'float64')
    for n in range(Qdim):
        for j in range(Qdim):
            dQ = get_dQ(j,qlist)                
            phi_integral = 0 
            for i in range(phidim):
                # round is added to avoid negative values
                q_diff = np.sqrt(np.round(qlist[n]**2+qlist[j]**2-2*qlist[n]*qlist[j]*np.cos(philist[i]),10))        
                if q_diff >= dQ:                    
                    phi_integral +=V(q_diff, epsilon_s_stat, d) # V_basic(q_diff, epsilon_s_stat)#            
            phi_integral *= dQ*qlist[j]*dphi
            phi_integral /= (2*np.pi)**2
            V_matrix[n,j] += phi_integral
    return V_matrix


##############################################################################################################################################################    
#utility functions
def partition_list(lst, n):
    """Divide a list into n equal (or approximately equal) parts."""
    # Calculate the size of each chunk
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

@jit(nopython = True)
def get_dQ(i, qlist):
    '''
    i: list index
    qlist: list of q-values
    returns: dQ
    '''
    if i > 0:
        return qlist[i]-qlist[i-1]
    else:
        return qlist[i]-0
    
@jit(nopython = True)
def fouriertrafo(x, omega_list, tlist):
    '''
    x: list of values x(t) to be transformed into frequency space representation x(w)
    omega_list: list of frequencies w 
    tlist: list of equally spaced time values t at which x(t) is given

    returns fouriertrafo x(w) of x(t)
    '''
    assert len(x) == len(tlist), "x and tlist must be of same length!"
    dt = tlist[1]-tlist[0] #assume equal t-spacing
    x_von_omega = np.zeros(len(omega_list), dtype = 'complex128')
    for j, omega in enumerate(omega_list):
        t_integral = 0 
        for i,t in enumerate(tlist):
            t_integral  += dt * np.exp(1j*omega*t)*x[i] #Vorzeichen exponentialfunktion andersrum als bei np.FFT
        x_von_omega[j] = t_integral
    return x_von_omega

def P(p, d_cv, qlist):
    '''
    p: 2d-array of microscopic polarizations p(q,t)
    d_cv: transition dipole moment
    qlist: list of q-values corresponding to p(q,t)

    returns: list of macroscopic polarizations P(t)
    '''
    Qdim, tdim = np.shape(p)
    #computes array with P values at all times
    P = np.zeros(tdim,dtype = 'complex128')
    for t in range(tdim):
        sum = 0
        for i in range(Qdim):
            dQ = get_dQ(i,qlist)
            sum += p[i][t]* qlist[i] * dQ /(2*np.pi)
        P[t] = sum
    return P*d_cv

@jit(nopython=True)
def P_t(p, d_cv, qlist): #(not integrated into P to reduce runtime)
    '''
    p: 1d-array of microscopic polarizations p(q) at specific time t
    d_cv: transition dipole moment
    qlist: list of q-values corresponding to p(q)

    returns: macroscopic polarization P at time t
    '''
    Qdim = len(qlist)
    P = 0
    for i in range(Qdim):
        dQ = get_dQ(i,qlist)
        P += p[i]* qlist[i] * dQ
    return P*d_cv /(2*np.pi)

def N(n, qlist):
    '''
    n: 2d-array of microscopic occupations n(q,t)
    qlist: list of q-values corresponding to n(q,t)

    returns: list of macroscopic occupations N(t)
    '''
    Qdim, tdim = np.shape(n)
    N = np.zeros(tdim,dtype = 'complex128')
    Qdim = len(qlist)
    for t in range(tdim):
        sum = 0
        for i in range(Qdim):
            dQ = get_dQ(i,qlist)
            sum += n[i][t]* qlist[i] * dQ /(2*np.pi)
        N[t] = sum
    return N



##############################################################################################################################################################    
#Multi quantum well 


def get_absorption_MQW(P_list, t_probe, tlist, omega_list, E_probe, sigma_probe, n_surrounding, n_medium, z_2, spacing, spacing_0, omega_0, reflections = False, pulse_shape = 'gauss', method = 'lambert_beer', fresnel = False):
    '''Compute absorption of MQW sample inside dielectric environment
    P_list: 2d-array of macroscopic polarizations P(n,t), where n denotes the index of the quantum well
    t_probe: central time of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    tlist: list of times t corresponding to P(n,t)
    omega_list: list of frequencies at which absorption is to be calculated
    E_probe: Field amplitude of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    sigma_probe: temporal standard deviation of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    n_surrounding: background refractive index outside of probe material
    n_medium: background refractive index inside probe material
    z_2: total length of probe material (background material for quantum wells e.g. GaAs)
    spacing: spacing between quantum wells
    spacing_0: spacing between air-material interface and first quantum well
    omega_0: carrier frequency of probe electric field. (Split off frequency in RWA)
    reflections (bool): If True, reflections at both air-material interfaces are taken ito account (inside and outside of material). If False, not reflections are taken into account
    pulse_shape: shape of probe field: 'gauss' or 'sigma'
    method: method for absorption calculation: 'lambert_beer' (as in experiment) or '1-R-T'
    '''
    n_wells = len(P_list)
    if not reflections:
        n_surrounding = n_medium
    
    P_ws = [fouriertrafo(P_n, omega_list=omega_list, tlist = tlist) for P_n in P_list]
    if pulse_shape == 'gauss':
        E0_t = gauss_pulse(tlist, E_probe, sigma_probe, t_probe)
        E0_w = fouriertrafo(E0_t,omega_list=omega_list,tlist=tlist)
    if pulse_shape == 'sigma':
        E0_t = sigma_pulse(tlist, E_probe, sigma_probe, t_probe)
        E0_w = fouriertrafo(E0_t,omega_list=omega_list,tlist=tlist)

    exp_fac = 1j*omega_0*(n_medium/c)
    if n_wells == 1:
        exp_fac = 0 #ignore position z_i, Because 1W spectra are taken with SQW implementation (delete line if MQW implementation is used for p(t))
    F = (n_surrounding+n_medium)**2*np.exp(-exp_fac*z_2) + (n_surrounding-n_medium)*(n_medium-n_surrounding)*np.exp(exp_fac*z_2) 
    A_back = 0
    C_for = 0

    sum_negative = 0
    sum_positive = 0
    for i in range(n_wells):
        sum_negative += P_ws[i]*np.exp(-exp_fac*(i*spacing+spacing_0)) 
    for i in range(n_wells):
        sum_positive += P_ws[i] * np.exp(exp_fac*(i*spacing+spacing_0))
        
    A_back += 1j*omega_0/(F*epsilon_0*c)*((n_medium-n_surrounding)*sum_negative * np.exp(exp_fac*z_2) + (n_medium+n_surrounding)*sum_positive * np.exp(-exp_fac*z_2))
    C_for += 4*n_surrounding*n_medium/F * E0_w * np.exp(-1j*omega_0/c*n_surrounding*z_2)
    C_for += 1j*omega_0/(F*epsilon_0*c)*((n_medium-n_surrounding)*sum_positive * np.exp(-1j*omega_0/c*n_surrounding*z_2) 
                                         + (n_medium+n_surrounding)*sum_negative * np.exp(-1j*omega_0/c*n_surrounding*z_2))
    
    if reflections:        
        A_back += (np.exp(-exp_fac*z_2)*(n_surrounding-n_medium)*(n_surrounding+n_medium) + np.exp(exp_fac*z_2)*(n_surrounding+n_medium)*(n_medium-n_surrounding))/F * E0_w

    if method == '1-R-T':
        if fresnel:
            absorption = 1- np.abs(A_back+(1-n_medium)/2*E0_w)**2/np.abs((1+n_medium)/2*E0_w)**2 - n_medium*np.abs(C_for)**2/np.abs((1+n_medium)/2*E0_w)**2
        else:
            absorption = 1- np.abs(A_back)**2/np.abs(E0_w)**2 - np.abs(C_for)**2/np.abs(E0_w)**2
    if method == 'lambert_beer':
        if reflections == True:
            field_through= 4*n_surrounding*n_medium/F * E0_w * np.exp(-1j*omega_0/c*n_surrounding*z_2)
            absorption = -np.log(np.abs(C_for)**2/np.abs(field_through)**2)
        
        else:
            absorption = -np.log(np.abs(C_for)**2/np.abs(E0_w)**2)

    return absorption
               

#time dynamics for single pulse
@jit(nopython = True)
def ddt_singlepulse_MQW(t,y, n_wells, d_cv, V_kq, omega_0, E_G, effective_mass, gamma_lattice, t_c , E_0, sigma, qlist, 
                        n_medium, n_surrounding, z_2, spacing = 28.6, spacing_0 = 4.6, reflections = False, pulse_shape = 'gauss'): 
    '''
    t: time t in current step
    y: 2d-array [[p1(q),n1(q)], ...,[px(q),nx(q)]] of microscopic polarizations p(q) and occupations n(q) for x quantum wells from last timestep
    n_wells: number of quantum wells
    d_cv: dipole transition matrix element
    V_kq: Coulomb matrix
    omega_0: central frequency of RWA for E-field and Bloch equations
    E_G: Energy gap of QW material
    effective mass: effective electron mass
    gamma_lattice: heuristic damping constant, representing electron-lattice interaction
    t_c = temporal center of electric field pulse
    E_0 = field amplitude of electric field pulse
    sigma = temporal standard deviation of electric field pulse
    qlist = list of impulse absolute values q corresponding to p(q), n(q)
    n_medium: background refractive index inside probe material
    n_surrounding: background refractive index outside of probe material
    z_2 = total width of GaAs sample
    spacing: spacing between quantum wells
    spacing_0: spacing between initial air-material interface and first quantum well
    reflections (bool): If True, reflections at both air-material interfaces are taken ito account (inside and outside of material). If False, no reflections are taken into account
    pulse_shape: shape of probe field: 'gauss' or 'sigma'

    returns: time derivative dy/dt at time t
    '''

    Qdim = len(qlist)
    p_lists = [y[i*Qdim:(i+1)*Qdim] for i in range(n_wells)] #polarizations of nth well correspond to p_lists[n] 
    n_lists = [y[(n_wells+i)*Qdim:(n_wells+i+1)*Qdim] for i in range(n_wells)] #occupations of nth well correspond to n_lists[n] 

    #Ableitungsvektoren initialisieren
    dp = [np.zeros(Qdim, dtype = 'complex128') for _ in range(n_wells)]
    dn = [np.zeros(Qdim, dtype = 'complex128') for _ in range(n_wells)]
    
    #E_0 zur Zeit t berechnen
    if pulse_shape == 'sigma':
        incident_field =  sigma_pulse(t, E_0=E_0, sigma = sigma, t_center = t_c)
    if pulse_shape == 'gauss':
        incident_field =  gauss_pulse(t, E_0=E_0, sigma = sigma, t_center = t_c)   
    P_list =[P_t(p_lists[i], d_cv, qlist) for i in range(n_wells)] 
    #equations of motion for Ps
    for n in range(n_wells):
        E_eff = effective_field(n, P_list , incident_field, omega_0, n_medium,n_surrounding,
                                 z_2, spacing, spacing_0, reflections) #Effektives E-Feld am n-ten QW
            #Diagonalbeiträge für P
        for i in range(Qdim):
            dp[n][i] -= 1j*(E_G/hbar - omega_0 + hbar * (0.5 / effective_mass) * (qlist[i]**2) - 1j*gamma_lattice)* p_lists[n][i] #Diagonalbeitrag
            for j in range(Qdim):
                dp[n][i]+= 2j/hbar*  V_kq[i][j] * n_lists[n][j] * p_lists[n][i] #n-Abhängigkeit 

        #Coulomb interaction und effektives E-Feld ergeben zusammen omega_R
            omega_R = 1/hbar* E_eff*d_cv             
            for j in range(Qdim):
                omega_R += 1/hbar*  V_kq[i][j] * p_lists[n][j]#Coulomb-interaction

        #Nebendiagonalen für p
            dp[n][i] -= 1j*omega_R * (2* n_lists[n][i] - 1)
        #Nebendiagonalen für n
            dn[n][i]  -= 2* np.imag(omega_R * np.conjugate(p_lists[n][i]))

    dp_flattened = np.array([item for sub_list in dp for item in sub_list])
    dn_flattened = np.array([item for sub_list in dn for item in sub_list])
    return np.concatenate((dp_flattened,dn_flattened))

@jit(nopython=True)
def effective_field(n, P_list, incident_field, omega_0, n_medium, n_surrounding, z_2, #eigen
                    spacing, spacing_0, reflections = False):
    '''
    n: index of quantum well where field should be evaluated
    P_list: list with macroscopic polarization of each quantum well  
    incident_field: incident E-field value at z_1 (before going into medium)  
    omega_0: central frequency of RWA for E-field and Bloch equations
    n_medium: background refractive index inside probe material
    n_surrounding: background refractive index outside of probe material
    z_2: total width of quantum well background medium
    spacing: spacing between wells
    spacing_0: distance between air->medium transition and first quantum well
    reflections (bool): If True, reflections at both air-material interfaces are taken ito account (inside and outside of material). If False, no reflections are taken into account
    '''
    if not reflections:
        n_surrounding = n_medium #homogenous background medium
    #initialize two terms for field coming from left- / right side
    sum_left = 0
    sum_right = 0
    n_wells = len(P_list)
    exp_fac = 1j*omega_0*(n_medium/c)
    # direct field contributions from quantum wells
    for i in range(n):
        sum_left += P_list[i]*np.exp(-exp_fac*(i*spacing+spacing_0)) 
    for i in range(n,n_wells):
        sum_right += P_list[i] * np.exp(exp_fac*(i*spacing+spacing_0))
    sum_left *= 1j*omega_0/(2*epsilon_0*n_medium*c)
    sum_right *= 1j*omega_0/(2*epsilon_0*n_medium*c)


    F = (n_surrounding+n_medium)**2*np.exp(-exp_fac*z_2) + (n_surrounding-n_medium)*(n_medium-n_surrounding)*np.exp(exp_fac*z_2) 
    #incident/reflected field contributions
    sum_left += 2*n_surrounding*(n_surrounding+n_medium)/F * incident_field * np.exp(-exp_fac*z_2)
    sum_right += 2*n_surrounding*(n_medium-n_surrounding)/F * incident_field * np.exp(exp_fac*z_2)

    #reflected qw contributions
    if reflections:
        sum_negative = 0 #sum with negative z_i exponent
        sum_positive = 0 #sum with positive z_i exponent
        for i in range(n_wells):
            sum_negative += P_list[i]*np.exp(-exp_fac*(i*spacing+spacing_0)) 
            sum_positive += P_list[i]*np.exp(exp_fac*(i*spacing+spacing_0)) 

        fac = 1j*omega_0*(n_medium-n_surrounding)/(2*epsilon_0*n_medium*c*F)
        sum_negative *=  fac
        sum_positive *=  fac

        sum_left +=(n_medium-n_surrounding)*sum_negative*np.exp(exp_fac*z_2) + (n_medium + n_surrounding) * sum_positive*np.exp(-exp_fac*z_2)#
        sum_right +=(n_medium+n_surrounding)*sum_negative*np.exp(exp_fac*z_2) + (n_medium - n_surrounding) * sum_positive*np.exp(exp_fac*z_2)# 
    #phase factors at qw position
    sum_left *= np.exp(exp_fac*(n*spacing+spacing_0)) 
    sum_right *=  np.exp(-exp_fac*(n*spacing+spacing_0)) 

    return sum_left + sum_right


##############################################################################################################################################################    
#single quantum well (for debugging and comparison)

#time dynamics for single pulse
@jit(nopython = True)
def gamma_R(p_list, qlist, d_cv, omega_0, n_ref): #RADIATIVE DAMPING nicht AUSGESCHALTET!!!
    #radiative damping for single quantum well
    Qdim = len(qlist)
    sum= 0
    for i in range(Qdim):
        dQ = get_dQ(i, qlist)
        sum +=p_list[i] *qlist[i] *dQ
    return sum *  d_cv**2 * (omega_0)/(2* epsilon_0*c*n_ref) *1/(2*np.pi)


@jit(nopython = True)
def ddt_singlepulse_SQW(t, y, V_kq, qlist, E_0, sigma, t_c, d_cv, omega_0, E_G, effective_mass, gamma_lattice, n_medium, pulse_shape = 'gauss'): #FOR SINGLE QUANTUM WELL!, No reflections taken into account
    '''  
    t: time t in current step
    y: 1d-array [p1(q),n1(q)] of microscopic polarizations p(q) and occupations n(q) from last timestep
    V_kq: Coulomb matrix
    qlist = list of impulse absolute values q corresponding to p(q), n(q)
    E_0 = field amplitude of electric field pulse
    sigma = temporal standard deviation of electric field pulse
    t_c = temporal center of electric field pulse
    d_cv: dipole transition matrix element
    omega_0: central frequency of RWA for E-field and Bloch equations
    E_G: Energy gap of QW material
    effective mass: effective electron mass
    gamma_lattice: heuristic damping constant, representing electron-lattice interaction
    pulse_shape: shape of probe field: 'gauss' or 'sigma'
    n_medium: background refractive index inside probe material

    returns: time derivative dy/dt at time t 
    '''

    #aufteilen der Zustands in p und n
    Qdim = len(qlist)
    p_list = y[:Qdim]
    n_list = y[Qdim:]

    #effektives Feld zur Zeit t berechnen
    if pulse_shape == 'sigma':
        E_eff = 1j*gamma_R(p_list,qlist,d_cv, omega_0, n_medium) + d_cv * sigma_pulse(t, E_0=E_0, sigma = sigma, t_center = t_c)
    if pulse_shape == 'gauss':
        E_eff = 1j*gamma_R(p_list,qlist,d_cv, omega_0, n_medium) + d_cv * gauss_pulse(t, E_0=E_0, sigma = sigma, t_center = t_c)
    #Ableitungsvektoren initialisieren
    dp = np.zeros(Qdim, dtype = 'complex128') 
    dn = np.zeros(Qdim, dtype = 'complex128') 

    #Diagonalbeiträge für P
    for i in range(Qdim):
        dp[i] -= 1j*(E_G/hbar - omega_0 + hbar * (0.5 / effective_mass) * (qlist[i]**2) - 1j*gamma_lattice)* p_list[i] #Diagonalbeitrag
        for j in range(Qdim):
            dp[i]+= 2j/hbar*  V_kq[i][j] * n_list[j] * p_list[i] #n-Abhängigkeit 

    #Coulomb interaction und effektives E-Feld ergeben zusammen omega_R
        omega_R = 0
        for j in range(Qdim):
            omega_R += 1/hbar*  V_kq[i][j] * p_list[j]#Coulomb-interaction
        omega_R += 1/hbar* E_eff #Effektives E-Feld
    
    #Nebendiagonalen für p
        dp[i] -= 1j*omega_R * (2* n_list[i] - 1)
      #  for j in range(Qdim): #only linear coulomb interaction, included in omega_R
        #    dp[i] += 1j* 1/hbar*  V_kq[i][j] * p_list[j]  #only linear coulomb interaction, included in omega_R
    #Nebendiagonalen für n
        dn[i] -= 2* np.imag(omega_R * np.conjugate(p_list[i]))
    return np.concatenate((dp,dn))

def get_absorption_SQW(P_list, t_probe, tlist,omega_list, E_probe, sigma_probe, omega_0, n_medium, pulse_shape = 'gauss', method = 'lambert_beer'):#FOR SINGLE QUANTUM WELL!, No reflections taken into account
    '''
    P_list: 1d-array of macroscopic polarization,0,0,0s P(t)
    t_probe: central time of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    tlist: list of times t corresponding to P(t)
    omega_list: list of frequencies at which absorption is to be calculated
    E_probe: Field amplitude of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    sigma_probe: temporal standard deviation of gaussian envelope for probe field. Coincides with pump field for single pulse spectra
    omega_0: carrier frequency of probe electric field. (Split off frequency in RWA)
    n_medium: refractive index of background medium
    pulse_shape: shape of probe field: 'gauss' or 'sigma'
    method: method for absorption calculation: 'lambert_beer' (as in experiment) or '1-R-T'
    '''
    P_w = fouriertrafo(P_list, omega_list=omega_list, tlist = tlist)
    if pulse_shape == 'gauss':
        E0_w = fouriertrafo(gauss_pulse(tlist, E_probe, sigma_probe, t_probe),omega_list=omega_list,tlist=tlist)
    if pulse_shape == 'sigma':
        E0_w = fouriertrafo(sigma_pulse(tlist, E_probe, sigma_probe, t_probe),omega_list=omega_list,tlist=tlist)
    E_R = 1j*omega_0/(2*epsilon_0*c*n_medium)*P_w
    R=np.absolute(E_R/E0_w)**2
    E_T = E0_w+E_R
    T=np.absolute(E_T/E0_w)**2
    if method == '1-R-T':
        return 1-R-T
    if method == 'lambert_beer':
        return -np.log(T)
    


##############################################################################################################################################################    
#field visualization

def reflection_transmission(E0_w, omega_0, n_surrounding, n_medium , z_2, reflections = True):
   #describes reflection and transmission of field when passing through GaAs block of length z_2 without quantum wells
    if not reflections:
        n_surrounding = n_medium
   # if pulse_shape == 'gauss':
    #    E0_t = gauss_pulse(tlist, E_probe, sigma_probe, t_probe)
     #   E0_w = fouriertrafo(E0_t,tlist=tlist,omega_list=omega_list)[325]
    #if pulse_shape == 'sigma':
     #   E0_t = sigma_pulse(tlist, E_probe, sigma_probe, t_probe)
      #  E0_w = fouriertrafo(E0_t,tlist=tlist,omega_list=omega_list)[325]
   
    exp_fac = 1j*omega_0*(n_medium/c)
    F = (n_surrounding+n_medium)**2*np.exp(-exp_fac*z_2) + (n_surrounding-n_medium)*(n_medium-n_surrounding)*np.exp(exp_fac*z_2) 

    
    C_for = 4*n_surrounding*n_medium/F * E0_w * np.exp(-1j*omega_0/c*n_surrounding*z_2)
    A_back = (np.exp(-exp_fac*z_2)*(n_surrounding-n_medium)*(n_surrounding+n_medium) + np.exp(exp_fac*z_2)*(n_surrounding+n_medium)*(n_medium-n_surrounding))/F * E0_w

    reflection = np.abs(A_back)**2/np.abs(E0_w)**2 
    transmission = np.abs(C_for)**2/np.abs(E0_w)**2
    #reflection = np.abs(A_back)/np.abs(E0_w)
   # transmission = np.abs(C_for)/np.abs(E0_w)
    
    return reflection, transmission

@jit(nopython=True)
def field_in_block(z, initial_field, omega_0, z_2, n_medium, n_surrounding, reflections = True):
#describes field at position z inside the GaAs structure of length z_2 without quantum wells
# p_vec: vector with all polarizations for each quantum well as entries (n,qdim)
# n: index of quantum well
# n_wells: number of quantum wells
# spacing_0: distance between air->medium transition and first quantum well
# spacing: spacing between wells
# z_2: width of quantum well background medium
# initial_field: incident E-field value at z_1 (transition air -> QW background medium)
# n_1: background refractive index of air
# n_2: background refractive index of qw background medium
    if not reflections:
        n_surrounding = n_medium #homogenous background medium
    #initialize two terms for field coming from left- / right side
    sum_left = 0
    sum_right = 0

    exp_fac = 1j*omega_0*(n_medium/c)

    F = (n_surrounding+n_medium)**2*np.exp(-exp_fac*z_2) + (n_surrounding-n_medium)*(n_medium-n_surrounding)*np.exp(exp_fac*z_2) 
    #incident/reflected field contributions
    sum_left += 2*n_surrounding*(n_surrounding+n_medium)/F * initial_field * np.exp(-exp_fac*z_2)
    sum_right += 2*n_surrounding*(n_medium-n_surrounding)/F * initial_field * np.exp(exp_fac*z_2)

    #phase factors at position z
    sum_left *= np.exp(exp_fac*(z)) 
    sum_right *=  np.exp(-exp_fac*(z)) 

    return sum_left + sum_right

##############################################################################################################################################################    
#equilibrium carrier population


@jit(nopython=True)
def equilibrium_occupation(k, chem_pot, T, E_G, mu):
    """
    Calculate then thermodynamic equilibrium occupation number of excitations with wave vector `k`.

    Parameters:
    -----------
    k : float
        Wave vector of the particle.
    chem_pot : float
        Chemical potential of the system.
    T : float
        Temperature of the system.
    E_G : float
        direct energy gap to the conduction band
    mu : float
        effective mass of excitation 

    Returns:
    --------
    float
        The equilibrium occupation number of the conduction band electron.
    """
    return 1/(np.exp((E_G + hbar**2 * k**2/(2*mu)-chem_pot)/(k_B*T))+1)


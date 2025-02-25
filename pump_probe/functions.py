from parameters_general import *
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
    power *= 6.24151*1e-3 #umrechnung uW in eV/fs
    A_probe = np.pi * (w_t) ** 2
    fluence = 1/(A_probe * gamma_rep) * (1-np.exp(-2* w_t**2/w_p**2))/(1-np.exp(-2)) * pf_corr * power 
    r = (n_ref-1)/(n_ref+1)
    if not substract_initial_reflection:
        r = 0
    return (1-r) * np.sqrt((4*np.sqrt(np.pi)*sigma)/(epsilon_0*c*n_ref) * fluence)

def E0_from_fluence(fluence, pf_corr, sigma, n_ref, substract_initial_reflection = True):
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

    fluence = fluence*6.24152*1e-2*pf_corr* 0.1503/0.2829 #umrechnung fluence in eV/nm^2 #umrechnung experiment theorie
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

def get_absorption_probe(P_list, t_probe, tlist,omega_list, E_probe, sigma_probe, omega_probe, n_medium, pulse_shape = 'gauss', method = 'lambert_beer'):#FOR SINGLE QUANTUM WELL!, No reflections taken into account
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
    E_R = 1j*omega_probe/(2*epsilon_0*c*n_medium)*P_w
    R=np.absolute(E_R/E0_w)**2
    E_T = E0_w+E_R
    T=np.absolute(E_T/E0_w)**2
    if method == '1-R-T':
        return 1-R-T
    if method == 'lambert_beer':
        return -np.log(T)
    

##############################################################################################################################################################    
#intraband relaxation

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


@jit(nopython=True)
def N_eq(chem_pot, T, qlist, Qrange, E_G, mu):
    """
    Calculate the thermodynamic equilibrium macroscopic particle number over a range of wave vectors `qlist`.

    Parameters:
    -----------
    chem_pot : float
        Chemical potential of the system.
    T : float
        Temperature of the system.
    qlist : array-like
        List of wave vectors 
    Qrange : int
        Range over which to compute the sum 
    E_G : float
        direct energy gap to the conduction band
    mu : float
        effective mass of excitation 

    Returns:
    --------
    float
        The total equilibrium particle number.
    """
    N_eq = 0
  #  Qdim = len(qlist)
    for i in range(Qrange):
        dQ = get_dQ(i,qlist)
        N_eq += equilibrium_occupation(qlist[i], chem_pot, T, E_G, mu)* qlist[i] * dQ /(2*np.pi)
    return N_eq

# @jit(nopython=True)
# def N_tot(qlist = qlist, Qrange = Q_eqrange):
#     """
#     Calculate the total particle number for a given wave vector list `qlist`.

#     Parameters:
#     -----------
#     qlist : array-like, optional
#         List of wave vectors (default is `qlist`).
#     Qrange : int, optional
#         Range over which to compute the sum (default is `Q_eqrange`).

#     Returns:
#     --------
#     float
#         The total particle number.
#     """
#     N_eq = 0
#     #Qdim = len(qlist)
#     for i in range(Qrange):
#         dQ = get_dQ(i,qlist)
#         N_eq += equilibrium_occupation(qlist[i], chem_pot, T)* qlist[i] * dQ * A/(2*np.pi)
#     return N_eq

@jit(nopython=True)
def Ekin_eq(chem_pot, T, qlist, Qrange, E_G, mu):
    """
    Calculate the thermodynamic equilibrium kinetic energy of the system.

    Parameters:
    -----------
    chem_pot : float
        Chemical potential of the system.
    T : float
        Temperature of the system.
    qlist : array-like, optional
        List of wave vectors (default is `qlist`).
    Qrange : int, optional
        Range over which to compute the sum (default is `Q_eqrange`).
    E_G : float
        direct energy gap to the conduction band
    mu : float
        effective mass of excitation 
    Returns:
    --------
    float
        The total equilibrium kinetic energy.
    """
    Ekin_eq = 0
    for i in range(Qrange):
        dQ = get_dQ(i,qlist)
        Ekin_eq += qlist[i]**3 *hbar**2 / (2*mu) * equilibrium_occupation(qlist[i], chem_pot, T, E_G,mu) * dQ /(2*np.pi)
    return Ekin_eq

@jit(nopython=True)
def Ekin(n_list, qlist,Qrange, mu):
    """
    Calculate the kinetic energy of the system based on a given occupation number list `n_list` at wave vectors 'qlist'.

    Parameters:
    -----------
    n_list : array-like
        List of occupation numbers corresponding to wave vectors in `qlist`.
    qlist : array-like
        List of wave vectors 
    Qrange : int
        Range over which to compute the sum 
    mu : float
        effective mass of excitation 
    Returns:
    --------
    float
        The total kinetic energy.
    """
    Ekin = 0
   # Qdim = len(qlist)
    for i in range(Qrange):
        dQ = get_dQ(i,qlist)
        Ekin += qlist[i]**3 *hbar**2 / (2*mu) * n_list[i] * dQ / (2*np.pi)
    return Ekin


@jit(nopython=True)
def gradient_descent_2d(func1, func2, target1, target2, x0, y0, qlist, eqrange, E_G, mu, learning_rate=100, tolerance=5e-7, max_iterations=1000000):
    """
    Performs 2D gradient descent to optimize two parameters (x, y) to match the target values for two given functions.

    This function attempts to minimize a custom cost function that combines the squared differences between two 
    functions (`func1` and `func2`) and their respective target values (`target1` and `target2`). It uses numerical 
    differentiation to compute gradients and iteratively updates the parameters (x, y) using gradient descent.

    Parameters:
    -----------
    func1 : function
        The first function to optimize. It should take parameters x, y, `qlist`, and `Qrange` and return a value.
    func2 : function
        The second function to optimize. It should take parameters x, y, `qlist`, and `Qrange` and return a value.
    target1 : float
        The target value that `func1` should approach.
    target2 : float
        The target value that `func2` should approach.
    x0 : float
        Initial guess for the x parameter.
    y0 : float
        Initial guess for the y parameter.
    qlist : array-like
        A list of wave vectors passed as arguments to `func1` and `func2`.
    eqrange : int
        The range of wave vectors over which the functions will be evaluated. passed as arguments to `func1` and `func2`.
    E_G : float
        direct energy gap to the conduction band passed as arguments to `func1` and `func2`.
    mu : float
        effective mass of excitation passed as arguments to `func1` and `func2`.
    learning_rate : float, optional
        The learning rate for gradient descent updates. Default is 100.
    tolerance : float, optional
        The tolerance for stopping the gradient descent. If the cost function value falls below this threshold,
        the optimization process terminates. Default is 5e-7.
    max_iterations : int, optional
        The maximum number of iterations allowed for gradient descent. Default is 1,000,000.

    Returns:
    --------
    x : float
        The optimized x parameter that minimizes the cost function.
    y : float
        The optimized y parameter that minimizes the cost function.
    
    Notes:
    ------
    - The cost function is a combination of squared errors from `func1` and `func2` relative to their targets. 
      The error from `func2` is weighted more heavily by a factor of 100 to emphasize its impact.
    - The numerical derivative is calculated using finite differences, with a small epsilon value for approximation.
    - If the cost function reaches a value below `tolerance`, the gradient descent terminates early.
    - The y parameter is updated with a significantly larger scaling factor (1,000,000) to account for sensitivity 
      differences between the two parameters.
    """
    def cost_function(x,y):
        return np.sqrt((func1(x, y, qlist = qlist, Qrange = eqrange, E_G = E_G, mu = mu) - target1)**2 +  (100*(func2(x, y, qlist = qlist, Qrange = eqrange, E_G = E_G, mu = mu) - target2))**2)
    
    # Numerical derivative function
    def numerical_derivative(x, y, epsilon=1e-6):
        return [(cost_function(x + epsilon, y) - cost_function(x, y)) / epsilon , (cost_function(x, y+epsilon) - cost_function(x, y)) / epsilon]

    # Initialize the parameter
    x = x0
    y = y0
    iterations = 0
    #xlist =[]
   # ylist = []  

    # Perform gradient descent
    while iterations < max_iterations:
        # Calculate the function value and its derivative at the current parameter
        cost_value = cost_function(x,y)
       # print(func_value)
        grad = numerical_derivative(x, y)
        #print(cost_value)


        # Check if the current function value is close enough to the target
        if abs(cost_value) < tolerance:
            break
        # Update the parameter using the gradient
        x -= learning_rate * grad[0]* cost_value 
        y -= 1000000*learning_rate* grad[1] * cost_value 

        iterations += 1
        #xlist.append(x)
       # ylist.append(y)


    return x,y#xlist, ylist, iterations

##############################################################################################################################################################    
#pump-probe dynamics


@jit(nopython = True)
def ddt_pump_probe(t,y, chem_pot, T, n0_list, V_kq, E_pump, t_pump, omega_pump, sigma_pump, E_probe, t_probe, omega_probe, sigma_probe, qlist, d_cv, Q_eqrange, mu, E_G, gamma_lattice, n_ref, intraband_rel = True, gamma_inter = 0.25*10**(-2), pump_detuning = 0): #WITHOUT P(-1,2)
    """
    Compute the time derivative of the density matrix components under a pump-probe experiment.

    This function models the dynamics of a system subjected to pump-probe pulses, accounting for intraband relaxation,
    Coulomb interactions, and external electric fields. It calculates the time evolution of particle populations (`n`)
    and polarization (`p`) in the system.

    Parameters:
    -----------
    t : float
        Time variable.
    y : array-like
         1d-array [p(q),n(q)] with lists of microscopic polarizations p(q) and occupations n(q) from last timestep
    chem_pot : float
        Chemical potential of the system.
    T : float
        Temperature of the system.
    n0_list : array-like
        Equilibrium occupation numbers for each wave vector.
    V_kq : 2D array
        Coulomb interaction matrix for the different wave vectors.
    E_pump : float
        Peak electric field of the pump pulse.
    t_probe : float
        Time when the probe pulse is applied.
    E_probe : float
        Peak electric field of the probe pulse.
    t_pump : float
        Time when the pump pulse is applied.
    sigma_pump : float
        Width of the pump pulse in time.
    sigma_probe : float
        Width of the probe pulse in time.
    qlist : array-like
        List of wave vectors.
    d_cv  : float
        dipole transition element for transition p
    Q_eqrange : int
        Range over which the Fermi distribution is evaluated to compute intraband carrier thermalization.
    mu : float
        effective mass of excitation 
    E_G : float
        direct energy gap to the conduction band
    gamma_lattice : float 
        heuristic damping constant, representing electron-lattice interaction
    n_ref : float
        refractive index of homogenous background medium
    intraband_rel : bool, optional
        Flag to enable/disable intraband relaxation. Default is True.
    gamma_inter : float (optional)
        relaxation rate for the relaxation towards a fermi occupation of the excited carriers n, defaults to 0.25*10**(-2)
    pump_detuning : float, optional
        Frequency detuning of the pump pulse. Default is 0.


    Returns:
    --------
    d_vec : array
        Time derivative of the density matrix components (both `p` and `n`).
    chem_pot : float
        Updated chemical potential after intraband relaxation.
    T : float
        Updated temperature after intraband relaxation.
    n0_list : array
        Updated equilibrium occupation numbers after intraband relaxation.
    """
    #order of ps: 2-1,10,01,-12
    #order of ns: 1-1,00,-11

    #FOR omega_pump != omega_probe THE EQUATIONS SHOULD BE DOUBLE CHECKED!

    Qdim = len(qlist)
    p_lists = [y[i*Qdim:(i+1)*Qdim] for i in range(3)]
    n_lists = [y[i*Qdim:(i+1)*Qdim] for i in range(3,5)]
    n_lists.append(np.conjugate(n_lists[0]))#n(-1,1) = n(1,-1)*
    #Ableitungsvektoren initialisieren
    dp = [np.zeros(Qdim, dtype = 'complex128') for l in range(3)]
    dn = [np.zeros(Qdim, dtype = 'complex128') for l in range(2)]
    N_tot = 0
    Ekin = 0
    #compute chemical potential for free carrier fermi distribution
   # global chem_pot #use the global variable and change it within the function NOT SUPPORTED IN NUMBA
    if intraband_rel: #t%100 >= 99: 
        for i in range(Q_eqrange):
            dQ = get_dQ(i, qlist)
            common_fac =  n_lists[1][i]* qlist[i] * dQ /(2*np.pi)
            N_tot += common_fac
            Ekin += qlist[i]**2 *hbar**2 / (2*mu) * common_fac
        N_tot = np.abs(N_tot)
        Ekin = np.abs(Ekin)
        if  np.abs(N_tot) > 1e-6:
            chem_pot, T  = gradient_descent_2d(N_eq, Ekin_eq, N_tot, Ekin, chem_pot, T,  qlist = qlist, eqrange = Q_eqrange, E_G = E_G, mu = mu)
            n0_list = equilibrium_occupation(qlist, chem_pot, T, E_G, mu)
            n0_list[np.isnan(n0_list)] = 0
            
    #equations of motion for Ps
    for dex in range(3): #indexdiagonal equations (ohne radiative damping)
        for i in range(Qdim):            
            for j in range(Qdim):
                dp[dex][i] -=  V_kq[i][j] * p_lists[dex][j] #p-p coulomb interaction # Q*dQ*2pi schon in V_kq enthalten
    
    pump_field = gauss_pulse(t, E_pump, sigma_pump, t_pump, pump_detuning)
    probe_field = gauss_pulse(t, E_probe, sigma_probe, t_probe)
    for i in range(Qdim): #different equations for each index
        dp[0][i] += (E_G - hbar*(2*omega_pump - omega_probe) + hbar**2 * (0.5 / mu) * (qlist[i]**2) - 1j*gamma_lattice*hbar)* p_lists[0][i] #diagonal equations
        dp[1][i] += (E_G - hbar*(omega_pump) + hbar**2 * (0.5 / mu) * (qlist[i]**2) - 1j*gamma_lattice*hbar)* p_lists[1][i] #diagonal equations
        dp[2][i] += (E_G - hbar*(omega_probe) + hbar**2 * (0.5 / mu) * (qlist[i]**2) - 1j*gamma_lattice*hbar)* p_lists[2][i] #diagonal equations

        #external E-field contributions
        dp[0][i] += 2* n_lists[0][i]*pump_field * d_cv 
        dp[1][i] += ((2* n_lists[1][i]-1)*pump_field+2* n_lists[0][i]*probe_field)  * d_cv 
        dp[2][i] += ((2* n_lists[2][i])*pump_field+ (-1)*probe_field)  * d_cv#(2* n_lists[1][i]-1)*probe_field)  * d_cv 
       # dp[3][i] += 2* n_lists[2][i]*E2(t,E_2,delta_t)  * d_cv 
        
       #equations of motion for ns:
        dn[0][i] += ((np.conjugate(pump_field)*p_lists[0][i] + np.conjugate(probe_field) * p_lists[1][i]) - (pump_field * np.conjugate(p_lists[2][i]))) * d_cv 
        dn[1][i] +=((np.conjugate(pump_field)*p_lists[1][i] + np.conjugate(probe_field) * p_lists[2][i]) - (pump_field * np.conjugate(p_lists[1][i]) + probe_field * np.conjugate(p_lists[2][i]))) * d_cv
        if (intraband_rel and N_tot> 1e-6):
            dn[1][i] -= hbar*1j*gamma_inter * (n_lists[1][i]- n0_list[i]) #decays to equilibrium occupation #equilibrium_occupation(qlist[i], chem_pot))
      #  dn[0][i] -= hbar*1j*gamma_n * (n_lists[1][i]) #decays to zero
       
        for j in range(Qdim):
            dQ = get_dQ(j,qlist)
            q_factor = dQ * qlist[j] /(2*np.pi) * d_cv**2/(epsilon_0*c*n_ref)
            #internal E-field contributions
            dp[0][i] += (1j*(2*omega_pump - omega_probe))* ((n_lists[1][i]-0.5)*p_lists[0][j] + n_lists[0][i]*p_lists[1][j]) *q_factor 
            dp[1][i] += (1j*omega_pump)* ((n_lists[1][i]-0.5)*p_lists[1][j] + n_lists[2][i]*p_lists[0][j] + n_lists[0][i]*p_lists[2][j]) *q_factor 
            dp[2][i] += (1j*omega_probe) * ((n_lists[1][i]-0.5)*p_lists[2][j] + n_lists[2][i]*p_lists[1][j]) *q_factor 

            # p-n Coulomb interaction

            dp[0][i] += 2*V_kq[i][j]* ((n_lists[1][i]*p_lists[0][j] + n_lists[0][i]*p_lists[1][j]) -  (n_lists[1][j]*p_lists[0][i] + n_lists[0][j]*p_lists[1][i]))
            dp[1][i] += 2*V_kq[i][j]* ((n_lists[1][i]*p_lists[1][j] + n_lists[2][i]*p_lists[0][j] + n_lists[0][i]*p_lists[2][j]) - (n_lists[1][j]*p_lists[1][i] + n_lists[2][j]*p_lists[0][i] + n_lists[0][j]*p_lists[2][i])) 
            dp[2][i] += 2*V_kq[i][j]* ((n_lists[1][i]*p_lists[2][j] + n_lists[2][i]*p_lists[1][j]) - (n_lists[1][j]*p_lists[2][i] + n_lists[2][j]*p_lists[1][i])) 
            
            #equations for n
            q_factor_2 = q_factor/2
            dn[0][i] += (V_kq[i][j] - 1j*omega_pump* q_factor_2)*(np.conjugate(p_lists[1][j])*p_lists[0][i] + np.conjugate(p_lists[2][j])*p_lists[1][i])
                                                                                                              
            dn[0][i] -= (V_kq[i][j] + 1j*omega_pump* q_factor_2)*(np.conjugate(p_lists[1][i])*p_lists[0][j] + np.conjugate(p_lists[2][i])*p_lists[1][j])
        
            dn[1][i] += (V_kq[i][j] - 1j*omega_pump* q_factor_2)*(np.conjugate(p_lists[0][j])*p_lists[0][i] + np.conjugate(p_lists[1][j])*p_lists[1][i] + np.conjugate(p_lists[2][j])*p_lists[2][i])
                                                                                                             
            dn[1][i] -= (V_kq[i][j] + 1j*omega_pump* q_factor_2)*(np.conjugate(p_lists[0][i])*p_lists[0][j] + np.conjugate(p_lists[1][i])*p_lists[1][j] + np.conjugate(p_lists[2][i])*p_lists[2][j])
        
            
    dp = np.array([item for sublist in dp for item in sublist])
    dn = np.array([item for sublist in dn for item in sublist])
    d_vec = np.concatenate((dp,dn))/(1j*hbar)
  #  d_real = d_vec.real
   # d_imag = d_vec.imag
    
    return d_vec, chem_pot, T, n0_list #np.concatenate((d_real,d_imag))



##############################################################################################################################################################    
#functions needed for automated spacial fourier transformation MISTAKE SOMEWHERE


# @jit(nopython = True)
# def generate_indices_n(m = m):
#     #indices for P must fullfill a+b = 0, |a|+|b|<= m
#     #only list indices with m_1 >= 0
#     pairs = []
#     for i in range(0, m+1):
#         for j in range(-m, 1):
#             if abs(i) + abs(j) <= m and i + j == 0:
#                 pairs.append((i,j))
#     return pairs

# @jit(nopython = True)
# def generate_indices_P(m = m):
#     #indices for P must fullfill a+b = 1, |a|+|b|<= m
#     pairs = []
#     for i in range(-m, m+1):
#         for j in range(-m, m+1):
#             if abs(i) + abs(j) <= m and i + j == 1:
#                 pairs.append((i,j))
#     return pairs

# def startvector_zero(m = m, Qdim = Qdim):
#     #returns starting vector corresponding to y(0) = 0
#     # m is the maximum order in the spacial fourier transform of n and P 
#     num_Ps = len(generate_indices_P(m))
#     num_ns = len(generate_indices_n(m)) # save compute by using n(a,-a) = n*(-a,a)
#     return np.zeros((num_Ps+num_ns) *Qdim, dtype = 'complex128')

# @jit(nopython = True)
# def ddt_nonlinear(t, y, E_1, E_2, delta_t): #IST FEHLERHAFT!!!
    
#     #erstelle listen mit index tupeln fuer n und p bis grad m
#     indices_p = generate_indices_P(m)
#     indices_n = generate_indices_n(m)

#     p_lists = [y[i*Qdim:(i+1)*Qdim] for i in range(len(indices_p))]
#     n_lists = [y[i*Qdim:(i+1)*Qdim] for i in range(len(indices_p),len(indices_p) + len(indices_n))]

#     #Ableitungsvektoren initialisieren
#     dp = [np.zeros(Qdim, dtype = 'complex128') for l in range(len(indices_p))]
#     dn = [np.zeros(Qdim, dtype = 'complex128') for l in range((len(indices_n)))]

#     #p-gleichungen
#     for dex in range(len(indices_p)):#schleife über index paare
        

#         for i in range(Qdim): #schleife über impulse q
#             #Beiträge mit gleichem k aber unterschiedlichem index (ext.E-Feld Beiträge)
#             if (indices_p[dex][0] == 1 and indices_p[dex][1] == 0):
#                 dp[dex][i] += 1j/hbar* E1(t,E_1) * d_cv 
#                 #print(str(indices_p[dex]) + ' 1j/hbar* E_1* d_cv')
#             if (indices_p[dex][1] == 1 and indices_p[dex][0] == 0):
#                 dp[dex][i] += 1j/hbar* E2(t,E_2, delta_t) *d_cv
#                # print(str(indices_p[dex]) + ' 1j/hbar* E_2*d_cv')
#             if indices_p[dex][0] >= 1:
#                 dp[dex][i] -= 2j/hbar*n_lists[indices_p[dex][0]-1][i] * E1(t, E_1) * d_cv
#                 #print(indices_p[dex], indices_n[indices_p[dex][0]-1], 'E1')
#             elif dex > 0:
#                # print(indices_p[dex], 'conjugate', indices_n[indices_p[dex][0]+1], 'E1')
#                 dp[dex][i] -= 2j/hbar*np.conjugate(n_lists[indices_p[dex][0]+1][i]) * E1(t,E_1) * d_cv            
#             if (indices_p[dex][1] <= 1 and dex < len(indices_p)-1):
#                 dp[dex][i] -= 2j/hbar*n_lists[-(indices_p[dex][1]-1)][i] * E2(t,E_2, delta_t)*d_cv
#                # print(indices_p[dex], indices_n[-(indices_p[dex][1]-1)], 'E2')
#             elif dex < (len(indices_p)-1):
#                 dp[dex][i] -= 2j/hbar*np.conjugate(n_lists[indices_p[dex][1]-1][i]) * E2(t,E_2, delta_t)*d_cv
#                 #print(indices_p[dex],'conjugate', indices_n[indices_p[dex][1]-1], 'E2'

#             dp[dex][i] -= 1j*(hbar * (0.5 / mu) * (qlist[i]**2) - 1j*gamma)* p_lists[dex][i] #Diagonalbeitrag

#             for j in range(Qdim):
#                 dp[dex][i] += 1j/hbar*  V_kq[i][j] * p_lists[dex][j] #p-p coulomb interaction # Q*dQ*2pi schon in V_kq enthalten

#                 dp[dex][i] -= omega_0/(hbar*2*epsilon_0*c*n_ref) *p_lists[dex][j] *dQ * qlist[j] * A/(2*np.pi) * d_cv**2#radiative damping indexdiagonaler beitrag
                
#                 for dex2 in range(len(indices_n)):
#                     #radiative damping nichtlinearer Beitrag
#                     diff1 = indices_p[dex][0] - indices_n[dex2][0]
#                     diff2 = indices_p[dex][1] - indices_n[dex2][1]

#                     if (np.abs(diff1) + np.abs(diff2)) <= m and diff1 + diff2 == 1:
#                         dp[dex][i] += omega_0/(hbar*epsilon_0*c*n_ref)*n_lists[dex2][i]*p_lists[indices_p.index((diff1,diff2))][j] * d_cv**2 * dQ * qlist[j] * A/(2*np.pi) #Radiative Damping Nichtlinear für m1 >= 0
#                         dp[dex][i] -= 2j/hbar * V_kq[i][j] * (n_lists[dex2][i]*p_lists[indices_p.index((diff1,diff2))][j] + n_lists[dex2][j]*p_lists[indices_p.index((diff1,diff2))][i]) #p-n Coulomb WW für m1 >= 0
                    
#                     if dex2 != 0:
#                         diff1 = indices_p[dex][0] + indices_n[dex2][0]
#                         diff2 = indices_p[dex][1] + indices_n[dex2][1]

#                         if (np.abs(diff1) + np.abs(diff2)) <= m and diff1 + diff2 == 1:
#                             dp[dex][i] += omega_0/(hbar*epsilon_0*c*n_ref)*np.conjugate(n_lists[dex2][i])*p_lists[indices_p.index((diff1,diff2))][j] * d_cv**2 * dQ * qlist[j] * A/(2*np.pi) #Radiative Damping Nichtlinear für m1 < 0
#                             dp[dex][i] -= 2j/hbar * V_kq[i][j] * (np.conjugate(n_lists[dex2][i])*p_lists[indices_p.index((diff1,diff2))][j] + np.conjugate(n_lists[dex2][j])*p_lists[indices_p.index((diff1,diff2))][i]) #p-n Coulomb WW für m1 < 0
            


#     #Beiträge für n        
#     for dex in range(len(indices_n)):
#         for i in range(Qdim):
#             #Beiträge mit gleichem k aber anderem index (ext- E-Feld Beiträge)
#             n1 = indices_n[dex][0]
#             n2 = indices_n[dex][1]
#             if np.abs(n1+1) + np.abs(n2) <= m and n1+1+n2 == 1:
#                 dn[dex][i] -= 1j/hbar * np.conjugate(E1(t,E_1)) * p_lists[indices_p.index((n1+1,n2))][i] * d_cv
#             if np.abs(n1) + np.abs(n2+1) <= m and n1+1+n2 == 1:
#                 dn[dex][i] -= 1j/hbar * np.conjugate(E2(t,E_2, delta_t)) * p_lists[indices_p.index((n1,n2+1))][i] * d_cv
#             if np.abs(1-n1) + np.abs(-n2) <= m and 1-n1-n2 == 1:
#                 dn[dex][i] += 1j/hbar * E1(t,E_1) * np.conjugate(p_lists[indices_p.index((1-n1,-n2))][i]) * d_cv
#             if np.abs(-n1) + np.abs(1-n2) <= m and 1-n1-n2 == 1:
#                 dn[dex][i] += 1j/hbar * E2(t,E_2, delta_t) * np.conjugate(p_lists[indices_p.index((-n1,1-n2))][i]) * d_cv
           
#             for j in range(Qdim):
                
#                 for dex2 in range(len(indices_p)):
#                     #radiative damping nichtlinearer Beitrag
#                     sum1 = indices_n[dex][0] + indices_p[dex2][0]
#                     sum2 = indices_n[dex][1] + indices_p[dex2][1]
#                    # diff1 = indices_p[dex2][0] - indices_n[dex][0]
#                    # diff2 = indices_p[dex2][1] - indices_n[dex][1]

#                     if (np.abs(sum1) + np.abs(sum2)) <= m and sum1 + sum2 == 1:
#                         dn[dex][i] -= omega_0/(2*hbar*epsilon_0*c*n_ref)*np.conjugate(p_lists[dex2][j])*p_lists[indices_p.index((sum1,sum2))][i] * d_cv * dQ * qlist[j] * A/(2*np.pi) #Radiative Damping Field contribution
#                         dn[dex][i] -= omega_0/(2*hbar*epsilon_0*c*n_ref)*np.conjugate(p_lists[dex2][i])*p_lists[indices_p.index((sum1,sum2))][j] * d_cv * dQ * qlist[j] * A/(2*np.pi) #Radiative Damping Field contribution
                                                
#                         dn[dex][i] -= 1j/hbar * V_kq[i][j] * (np.conjugate(p_lists[dex2][j])*p_lists[indices_p.index((sum1,sum2))][i] )#p-p Coulomb WW #Summenfaktoren bereits in V_kq
#                         dn[dex][i] += 1j/hbar * V_kq[i][j] *( np.conjugate(p_lists[dex2][i])*p_lists[indices_p.index((sum1,sum2))][j])#p-p Coulomb WW
#                     #if (np.abs(diff1) + np.abs(diff2)) <= m and diff1 + diff2 == 1:
#                       #  dn[dex][i] += 1j/hbar * V_kq[i][j] *( p_lists[dex2][j]*np.conjugate(p_lists[indices_p.index((diff1,diff2))][i]))#p-p Coulomb WW

    
#     dp = np.array([item for sublist in dp for item in sublist])
#     dn = np.array([item for sublist in dn for item in sublist])

#     return np.concatenate((dp,dn))

##############################################################################################################################################################    
#deprecated functions
# @jit(nopython=True)
# def gradient_descent_1d(func, target, initial_param,learning_rate=10, tolerance=1e-8, max_iterations=100000):
#     """
#     Performs gradient descent to optimize the parameter of the given function to return the target value.
    
#     Parameters:
#     -----------
#         func (function): The function to optimize. Should take a single argument.
#         target (float): The target value that we want the function to return.
#         initial_param (float): The initial guess for the parameter.
#         learning_rate (float): The learning rate for gradient descent. Default is 0.01.
#         tolerance (float): The tolerance for stopping the gradient descent. Default is 1e-6.
#         max_iterations (int): The maximum number of iterations. Default is 1000.
        
#     Returns:
#     --------
#         float: The optimized parameter.
#         (int: The number of iterations performed.) only if activated
#     """
#     # Numerical derivative function
#     def numerical_derivative(func, x, epsilon=1e-6):
#         return (func(x + epsilon) - func(x)) / epsilon

#     # Initialize the parameter
#     param = initial_param
#     iterations = 0

#     # Perform gradient descent
#     while iterations < max_iterations:
#         # Calculate the function value and its derivative at the current parameter
#         func_value = func(param)
#        # print(func_value)
#         grad = numerical_derivative(func, param)


#         # Check if the current function value is close enough to the target
#         if abs(func_value - target) < tolerance:
#             break
#         # Update the parameter using the gradient
#         param -= learning_rate * (func_value - target) * grad
#         iterations += 1
#        # print(iterations)
        

#     return param#, iterations
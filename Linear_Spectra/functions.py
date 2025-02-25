from parameters_general import *
from numba import jit


##############################################################################################################################################################    
#Coulomb interaction

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

##############################################################################################################################################################    
#Exciton picture

@jit(nopython = True)
def wannier_matrix(qlist, philist, m_eff, epsilon_s_stat, d): 
    """
    Generates the matrix describing the eigenvalue problem for the Wannier equation

    Parameters:
    -----------
    qlist : list or array-like
        A list of values representing the momentum grid in the q-dimension, assumed equidistant.
    philist : list or array-like
        A list of values for the angular component (phi), also assumed equidistant.
    m_eff : float
        The effective mass of the particle, e.g. averaged effective electron and hole mass.
    epsilon_s_stat: static dielectric constant of background material
    d: effective quantum well thickness
    Returns:
    --------
    matrix : ndarray of complex numbers
        A complex matrix of shape (len(qlist), len(qlist)) where each element represents the 
        contribution to the Wannier eigenvalue problem at the respective (q, q') values.
    
    Notes:
    ------
    - The matrix includes both kinetic and potential energy contributions.
    - Assumes that qlist and philist are equidistant grids for efficient computation.
    - The potential V is applied radially based on the relative distance in momentum space, 
      calculated from qlist and philist.
    """
    Qdim = len(qlist)
    phidim = len(philist)
    dQ = qlist[1]-qlist[0] 
    dphi = philist[1]-philist[0]
    matrix = np.zeros((Qdim,Qdim), dtype='complex_')
    for n in range(Qdim):
        for j in range(Qdim):
            if n == j:
                matrix[n][j] += hbar * hbar * 0.5 / m_eff * qlist[n]**2
            # every matrix element has this contribution
            phi_integral = 0    
            for i in range(phidim):
                # round is added to avoid negative values
                q_absolute = np.sqrt(np.round(qlist[n]**2+qlist[j]**2-2*qlist[n]*qlist[j]*np.cos(philist[i]),10))
                if q_absolute > dQ:
                    phi_integral += V(q_absolute, epsilon_s_stat, d)      
            phi_integral *= dQ*qlist[j]*dphi
            phi_integral /= (2*np.pi)**2
            matrix[n][j] -= phi_integral    
    return matrix


def solve(matrix):
    """
    Diagonalizes a given matrix and returns its eigenvalues and eigenvectors.

    Parameters:
    -----------
    matrix : ndarray
        A square matrix (n x n) to be diagonalized, with complex or real values.

    Returns:
    --------
    EWs : ndarray
        An array of eigenvalues of the matrix.
    EVs : ndarray
        A matrix (n x n) where each column is an eigenvector corresponding to the respective 
        eigenvalue in `EWs`.
    
    Notes:
    ------
    - The function outputs the number of eigenvalues and eigenvectors found.
    - The matrix diagonalization is done using `numpy.linalg.eig`.
    """
    EWs, EVs = np.linalg.eig(matrix)
    #print("Es wurden {EW} Eigenwerte und {EV} Eigenvektoren gefunden.".format(EW=len(EWs), EV=len(EVs)))
    negativcounter = 0
    for EW in EWs:
        if EW.real < 0:
            negativcounter += 1
    #print("davon haben {neg} negativen Realteil".format(neg=negativcounter))
    return EWs, EVs

@jit(nopython = True)
def normalize(EVs, qlist):
    """
    Normalizes the eigenvectors of a system based on a given momentum grid.

    Parameters:
    -----------
    EVs : ndarray
        A matrix where each column represents an eigenvector that will be normalized.
    qlist : list or array-like
        A list of momentum values in the q-dimension, assumed to be equidistant.

    Returns:
    --------
    EVs : ndarray
        The matrix of eigenvectors, with each column normalized according to the qlist grid.

    Notes:
    ------
    - The function assumes an equidistant momentum grid (`qlist`) to compute a consistent normalization.
    - Each eigenvector is scaled by the square root of its total area, using a weighting factor of 
      q*dQ/(2*pi), where `dQ` is the spacing between adjacent points in `qlist`.
    - Eigenvectors with a zero normalization area are left unchanged.
    """
    dQ = qlist[1]-qlist[0] 
    for i in range(len(EVs)):
        area = 0.
        for j in range(len(EVs)):
            q = qlist[j] 
            area += q*dQ/2/np.pi*np.conjugate(EVs[j,i])*EVs[j,i]
        for j in range(len(EVs)):
            if area != 0:
                EVs[j,i]/= np.sqrt(area)
    return EVs


def sort(EWs,EVs):
    """
    Sorts eigenvalues and their corresponding eigenvectors in ascending order of eigenvalues.

    Parameters:
    -----------
    EWs : ndarray
        An array of eigenvalues.
    EVs : ndarray
        A matrix where each column is an eigenvector corresponding to the respective eigenvalue in `EWs`.

    Returns:
    --------
    EWs : ndarray
        The sorted array of eigenvalues in ascending order.
    EVs : ndarray
        The matrix of eigenvectors, rearranged so that each column corresponds to the sorted eigenvalues.

    Notes:
    ------
    - Sorting is performed in-place, rearranging `EWs` and `EVs` so that the eigenvalues in `EWs`
      are in ascending order, with corresponding columns in `EVs`.
    - Assumes that `EWs` and `EVs` have compatible shapes, where each column of `EVs` is an eigenvector.
    """
    Qdim = len(EWs)
    for i in range(Qdim):
        for j in range(i,Qdim):
            if (EWs[j]<EWs[i]):
                EWs[[i,j]] = EWs[[j,i]]
                EVs[:,[i, j]] = EVs[:,[j, i]]
    return EWs, EVs


def solve_(matrix, qlist):
    """
    Wrapper function to solve, normalize, and sort the eigenvalues and eigenvectors of a given matrix.

    This function performs the following steps:
    1. Diagonalizes the matrix to find its eigenvalues and eigenvectors.
    2. Normalizes the eigenvectors.
    3. Sorts the eigenvalues in ascending order and arranges eigenvectors accordingly.

    Parameters:
    -----------
    matrix : ndarray
        The matrix to be diagonalized, normalized, and sorted.
    qlist : np.array
        momentum grid for eigenvector normalization

    Returns:
    --------
    EWs : ndarray
        The sorted array of eigenvalues.
    EVs : ndarray
        The matrix of normalized and sorted eigenvectors, where each column corresponds to an eigenvalue in `EWs`.

    Notes:
    ------
    - This wrapper function utilizes `solve`, `normalize`, and `sort` to process the eigensystem in a streamlined manner.
    """
    EWs, EVs = solve(matrix)
    EVs = normalize(EVs, qlist)
    EWs, EVs = sort(EWs,EVs)
    return EWs, EVs

def phi0(EVs,mu, qlist):
    """
    Calculates the exciton orbital amplitude for a specific eigenstate `mu` at `r = 0`.

    Parameters:
    -----------
    EVs : ndarray
        A matrix of eigenvectors, where each column represents an eigenstate, with each row corresponding 
        to a specific momentum value in `qlist`.
    mu : int
        The index of the eigenstate (column in `EVs`) for which the exciton orbital value at `r = 0` 
        is calculated.
    qlist : list or array-like
        A list of momentum values in the q-dimension, assumed to be equidistant.

    Returns:
    --------
    phi0 : float
        The calculated exciton orbital amplitude for the specified eigenstate `mu` at `r = 0`.

    Notes:
    ------
    - The function uses a weighted sum over momentum values in `qlist` to compute `phi0`, which represents
      the amplitude of the exciton orbital at the origin (`r = 0`) in the specified eigenstate.
    - Assumes `qlist` has equidistant spacing, calculated as `dQ = qlist[1] - qlist[0]`.
    - The calculation includes a factor of `q * dQ / (2 * pi)` for each momentum component.
    """

    phi0 = 0.
    Qdim = len(qlist)
    dQ = qlist[1]-qlist[0]
    for j in range(Qdim):
        q = qlist[j] 
        phi0 += q*dQ/2/np.pi*EVs[j,mu]    
    return phi0   

def chi(EWs,EVs, mu, omega, omega_0, d_cv, gamma, n_ref, qlist):
    return -d_cv**2 * np.abs(phi0(EVs,mu, qlist))**2*(1/(hbar*((omega)-EWs[mu]/hbar + 1j*gamma + 1j* d_cv**2 * np.abs(phi0(EVs,mu, qlist))**2 *omega_0/(2*hbar*c*n_ref*epsilon_0))))/epsilon_0#+1/(hbar*(-omega-E_G/hbar-EWs[mu]/hbar - 1j*gamma - 1j*d_cv**2 * np.abs(phi0(EVs,mu))**2 *omega/(epsilon_0*c*n_ref))))
        
def E_R_mu(EWs,EVs, mu, omega_0, n_ref, omega_list, d_cv, gamma, qlist):
    '''E_R/E_0, emitted field per incoming field'''
    E_R = []
    for omega in omega_list:    
        E_R.append(1j*omega_0/(2*epsilon_0 * c*n_ref)* epsilon_0*chi(EWs,EVs,mu,omega, omega_0, d_cv, gamma, n_ref, qlist))
    return E_R

def E_R_full(EWs,EVs, omega_0, n_ref, omega_list, d_cv, gamma, qlist):
    energyresolution = len(omega_list)
    E_R = np.zeros(energyresolution,dtype = 'complex128')
    for mu in range(len(EWs)):
        E_R = E_R + np.array(E_R_mu(EWs,EVs, mu, omega_0, n_ref, omega_list, d_cv, gamma, qlist))
    return E_R

def alpha_exciton(mu, omega, EWs, EVs, d_cv, gamma, omega_0, n_ref, qlist):
    gamma_R =  d_cv**2 * np.abs(phi0(EVs,mu, qlist))**2 *(omega_0+omega)/(2*hbar*c*n_ref*epsilon_0)
    alpha = d_cv**2 * np.abs(phi0(EVs,mu, qlist))**2 *(omega_0+omega)/(c*n_ref) * (gamma + gamma_R)/(hbar*(omega-EWs[mu]/hbar)**2+hbar*(gamma+gamma_R)**2)/epsilon_0
    return alpha
    #return (omega_0)/(n_ref*c) * np.imag(chi(EWs,EVs, mu, omega, omega_0, d_cv, gamma, n_ref, qlist)) - omega_0**2/(n_ref*c)**2/2 *np.abs(chi(EWs,EVs, mu, omega, omega_0, d_cv, gamma, n_ref, qlist))**2

def get_absorption(E_R, energyresolution):
    R = np.absolute(E_R)**2
    T = np.absolute(np.ones(energyresolution, dtype = 'complex_')+E_R) **2
    return np.ones(energyresolution)- R - T

##############################################################################################################################################################    
#e-h picture

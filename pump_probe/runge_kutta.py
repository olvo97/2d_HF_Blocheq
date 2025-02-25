import numpy as np 
from numba import jit



@jit(nopython = True)
def runge_kutta_4(system, t, y, dt, chem_pot, T, n0_list):
    """
    Perform one step of the fourth-order Runge-Kutta integration.
    
    Parameters:
    system : function
        Function defining the coupled system of differential equations.
    y : numpy.ndarray
        Current state of the system (vector of complex variables).
    t : float
        Current time.
    dt : float
        Time step size.
    
    Returns:
    numpy.ndarray
        Updated state of the system after one Runge-Kutta step.
    """

    k1, chem_pot, T, n0_list = system(t, y, chem_pot, T, n0_list)
    k1*=dt
    #chem_pot = dt * system(t, y, chem_pot, n0_list)[1]
   # n0_list = dt * system(t, y, chem_pot, n0_list)[0]
    k2 = dt * system(t + 0.5 * dt, y + 0.5 * k1, chem_pot, T, n0_list)[0]
    k3 = dt * system(t + 0.5 * dt, y + 0.5 * k2, chem_pot, T, n0_list)[0]
    k4 = dt * system(t + dt, y + k3, chem_pot, T, n0_list)[0]
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6 , chem_pot, T, n0_list

@jit(nopython = True)
def solve_runge_kutta(system, t_list, y0, chem_pot, T, n0_list):
    #system: system of differential equations, dependent on state vector y and time t
    # Array to store solutions
    y = np.zeros((len(t_list), len(y0)), dtype='complex128')
    y[0] = y0
    dt = t_list[1]-t_list[0]
    pot_list = np.zeros(len(t_list))
    T_list = np.zeros(len(t_list))
    pot_list[0] = chem_pot
    T_list[0] = T

    # Time integration
    for i in range(1, len(t_list)):
        y[i], chem_pot, T, n0_list = runge_kutta_4(system, t_list[i-1], y[i-1], dt, chem_pot, T, n0_list)
        pot_list[i] = chem_pot
        T_list[i] = T
    return y, pot_list, T_list, n0_list

import numpy as np
from TrapAnalysis import artificial_anneal as anneal

def get_resonator_constants():
    """
    Returns a dictionary of resonator constants used in the calculations in this module.
    :return: {f0, Z0, Q, input power}
    """
    constants = {'f0': 6.405E9, 'Z0': 90.0, 'Q': 18000, 'P': -100, 'l': 4500E-6}
    return constants

def get_physical_constants():
    """
    Returns a dictionary of physical constants used in the calculations in this module.
    :return: {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    """
    constants = {'e': 1.602E-19, 'm_e': 9.11E-31, 'eps0': 8.85E-12, 'hbar': 1.055E-34}
    return constants

def get_V0():
    """
    Returns the single photon voltage
    :return: V0 in V
    """
    c = get_physical_constants()
    r = get_resonator_constants()
    return c['hbar'] * 2 * np.pi * r['f0'] / c['e']

def get_z0(electron_frequency):
    """
    Returns the electron motion quantum in m
    :param electron_frequency: frequency in Hz
    :return: z0 in m
    """
    c = get_physical_constants()
    return np.sqrt(c['hbar'] / (2 * c['m_e'] * 2 * np.pi * electron_frequency))

def get_g(electron_frequency, Efield):
    """
    This returns the coupling in units of radians: g[rad] = 2*pi*g[Hz]
    :param electron_frequency: electron frequency in Hz
    :param Efield: Electric field evaluated at a single point in space, with 1V applied to the electrodes
    :return: g[rad] = 2 * pi * g[Hz]
    """
    c = get_physical_constants()
    V0 = get_V0()
    z0 = get_z0(electron_frequency)
    return c['e'] * V0 * Efield * z0 / c['hbar']

def calculate_metrics(xi, yi, box_y_length=40E-6, eps=1E-15):
    Xi, Yi = np.meshgrid(xi, yi)
    Xj, Yj = Xi.T, Yi.T

    Yi_shifted = Yi.copy()
    Yi_shifted[Yi_shifted > 0] -= box_y_length  # Shift entire box length
    Yj_shifted = Yi_shifted.T
    XiXj = Xi - Xj
    YiYj = Yi - Yj
    YiYj_shifted = Yi_shifted - Yj_shifted

    Rij_standard = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
    Rij_shifted = np.sqrt((XiXj) ** 2 + (YiYj_shifted) ** 2)
    R_ij = np.minimum(Rij_standard, Rij_shifted)

    # Avoid an raising a warning when dividing by Xi-Xj in theta_ij
    np.fill_diagonal(XiXj, eps)

    theta_ij = np.arctan(YiYj / XiXj)
    theta_ij_shifted = np.arctan(YiYj_shifted / XiXj)

    # Use shifted y-coordinate only in this case:
    np.copyto(theta_ij, theta_ij_shifted, where=Rij_shifted < Rij_standard)

    # Remember to set the diagonal back to 0
    np.fill_diagonal(theta_ij, 0)

    return R_ij, theta_ij

def setup_eom(electron_positions, Vres, x_symmetric, Uinterp_symmetric, Uinterp,
              eps=1E-15, screening_length=np.inf):
    """
    Sets up the inverse matrix that needs to be solved
    :param electron_positions: coordinates of the electron configuration = [x0, y0, x1, y1, ...]
    :param Vres: float, resonator voltage.
    :return: M^(-1) * K
    """
    c = get_physical_constants()
    r = get_resonator_constants()

    omega0 = 2 * np.pi * r['f0']
    L = r['Z0'] / omega0
    C = 1 / (omega0 ** 2 * L)

    RS = anneal.ResonatorSolver(x_symmetric, -Vres * Uinterp_symmetric, efield_data=Uinterp[0])

    num_electrons = np.int(len(electron_positions) / 2)
    xi, yi = anneal.r2xy(electron_positions)

    # Set up the inverse of the mass matrix first
    diag_invM = 1 / c['m_e'] * np.ones(num_electrons + 1)
    diag_invM[0] = 1 / L
    invM = np.diag(diag_invM)

    # Set up the kinetic matrix next
    K = np.zeros(np.shape(invM))
    K[0, 0] = 1 / C  # Bare cavity
    # Note this should be the first derivative of the RF potential,
    # and not the DC potential
    K[1:, 0] = K[0, 1:] = c['e'] / C * RS.Ex(xi, yi)

    Rij, Tij = calculate_metrics(xi, yi, eps=eps)
    np.fill_diagonal(Rij, eps)
    if screening_length == np.inf:
        # Note that an infinite screening length corresponds to the Coulomb case. Usually it should be twice the
        # helium depth
        kij_plus_matrix = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * (1 + 3 * np.cos(2 * Tij)) / Rij ** 3
    else:
        Rij_scaled = Rij/screening_length
        kij_plus_matrix = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * np.exp(-Rij_scaled) / Rij**3 * \
                          (1 + Rij_scaled + Rij_scaled**2 + (3 + 3*Rij_scaled + Rij_scaled**2) * np.cos(2 * Tij))

    np.fill_diagonal(kij_plus_matrix, 0)
    kij_plus_sum = np.sum(kij_plus_matrix, axis=1)

    # Note: added - sign for c['e']
    K_block = -kij_plus_matrix + np.diag(c['e'] * RS.ddVdx(xi, yi) + kij_plus_sum)
    K[1:, 1:] = K_block

    return np.dot(invM, K)

def solve_eom(LHS):
    """
    Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
    :param LHS: matrix product of M^(-1) K
    :return: Eigenvalues, Eigenvectors
    """
    EVals, EVecs = np.linalg.eig(LHS)
    return EVals, EVecs

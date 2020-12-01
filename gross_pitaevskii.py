import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

EPS = 1e-12

def TSSP_1d(m, N, a, b, psi0, potential, dt, k, eps):

    grid_points = 2**m

    psi = np.empty((N, grid_points), dtype=complex)

    x = np.linspace(a, b, grid_points, endpoint=False)
    t = np.linspace(0, N*dt, N, endpoint=False)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)

    V = potential(x)

    psi[0,:] = psi0(x)

    for n in range(N-1):
        psi[n+1,:] = timeindep_tssp_1d_step(psi[n,:], V)

    return t, x, psi



def timeindep_tssp_1d_step(psi, V):

    p1 = psi * np.exp(-1j*(V + k * np.abs(psi)**2) * dt/(2*eps))
    p2 = fft.ifft(np.exp(-1j*eps*dt*mu**2/2) * fft.fft(p1))
    return p2 * np.exp(-1j*(V + k * np.abs(p2)**2)*dt/(2*eps))



def TSSP_2d(m, time_steps, a, b, psi0, potential, dt, k1, eps, i_coeff=False):

    grid_points = 2**m
    psi = np.empty((time_steps + 1, grid_points, grid_points), dtype=complex)

    x = np.linspace(a, b, grid_points)
    y = np.linspace(a, b, grid_points)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = np.linspace(0, time_steps*dt, time_steps + 1)

    V = potential(X, Y)

    f = 2*np.pi * fft.fftfreq(grid_points, d = 2*np.pi/grid_points)
    Fx, Fy = np.meshgrid(f, f, sparse=False, indexing="ij")

    psi[0,:] = psi0(X, Y)

    if i_coeff:
        for i in range(time_steps + 1):
            psi[i+1,:] = timeindep_tssp_2d_step(psi[i,:], k1, eps, V, Fx, Fy)
    else:
        for i in range(time_steps + 1):
            psi[i+1,:] = timedep_tssp_2d_step(psi[i,:], k1, eps, V, Fx, Fy)

    return t, X, Y, psi



def timeindep_tssp_2d_step(psi, k1, eps, V, Fx, Fy):

    psi1 = ps * np.exp(-1j*(V + k1*np.abs(psi)**2)*dt/(2*eps))
    psihat1 = fft.fft2(psi1)
    psihat2 = psihat1 * np.exp(-1j* eps*dt * 4*np.pi**2*(Fx**2 + Fy**2)/(b-a)**2)
    psi2 = fft.ifft2(psihat2)
    return psi2 * np.exp(-1j*(V + k1*np.abs(psi2)**2)*dt/(2*eps))



def timedep_tssp_2d_step(psi, k1, eps, V, Fx, Fy):
    #to be changed
    psi1 = ps * np.exp(-1j*(V + k1*np.abs(psi)**2)*dt/(2*eps))
    psihat1 = fft.fft2(psi1)
    psihat2 = psihat1 * np.exp(-1j* eps*dt * 4*np.pi**2*(Fx**2 + Fy**2)/(b-a)**2)
    psi2 = fft.ifft2(psihat2)
    return psi2 * np.exp(-1j*(V + k1*np.abs(psi2)**2)*dt/(2*eps))


'''
def timedep_gp_1d(m, time_steps, a, b, psi0, beta, dt, potential):

    grid_points = 2**m
    psi = np.empty((time_steps + 1, grid_points), dtype=complex)

    x = np.linspace(a, b, grid_points)
    t = np.linspace(0, time_steps*dt, time_steps + 1)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)

    V = potential(x)
    expV = np.exp(-k*V)
    zero_pot = V == 0

    psi[0,:] = psi0(x)

    for i in range(time_steps + 1):
        psi[i+1,:] = timedep_tssp_1d_step(psi[i,:], V, expV, k, \
                                            beta, mu_l, zero_pot)

    return t, x, psi



def timedep_tssp_1d_step(psi, V, expV, k, beta, mu_l, zero_pot):

    abs_psi = np.abs(psi)**2
    p1 = psi * np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_psi))
    p1[zero_pot] = psi * 1 / np.sqrt(1 + beta*k*abs_psi)

    p2 = 1

    abs_p2 = np.abs(p2)**2
    p3 = p2 * np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_p2))
    p3[zero_pot] = p2 * 1 / np.sqrt(1 + beta*k*abs_p2)

    return p3 / np.abs(p3)
'''


def mean_value(f, psi, a, b, M):
    """
    Return the mean value of a function evaluated on a square grid of size
    (b-a)^2 with M^2 points on it w.r.t. the probability density defined by
    the wavefunction psi.
    ...

    Parameters
    ----------
    f : function, 2 arguments
        The function of which calculate the mean value. First argument should
        be the x coordinate and the second the y coordinate.
    psi : numpy matrix
        Wave function that defines the probability deensity.
    a : float
        Beginning of the two axes.
    b : float
        Ending of the two axes.
    M : int
        Number of points per axis.

    Raises
    ------
    RuntimeError
        If the size of psi don't match with the gird of paramethes a, b and M.

    Returns
    -------
    scalar
        mean value of the function f.

    """
    delta = np.abs(b-a)/M
    #this is the area element for the case of a square grid. a,b represent the limits of the square in 2D
    dA = delta**2

    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")

    if psi.shape == x.shape:
        return np.sum(f(X,Y) * np.abs(psi)**2 * dA)
    else:
        raise RuntimeError("Size of psi should match the size of the grid.")

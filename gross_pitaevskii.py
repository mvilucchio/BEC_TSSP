import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

EPS = 1e-12

def freq_maker(N):
    """
    Wrapper for the creation of frequencies used in the following spectral method.
    ...

    Parameters
    ----------
    N : integer
        Number of division on the grid

    Returns
    -------
    ndarray
        Frequencies generated by fourier transform.
    """
    return 2 * np.pi * fft.fftfreq(N, d = 2*np.pi/N)



def TSSP_1d(m, N, a, b, psi0, potential, dt, k, eps):

    grid_points = 2**m

    psi = np.empty((N, grid_points), dtype=complex)

    x = np.linspace(0, 1, grid_points, endpoint=False)
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



def TSSP_2d(M, N, a, b, psi0, dt, k1, eps):

    psi = np.empty((N, M, M), dtype=complex)

    x = np.linspace(a, b, M, endpoint=False)
    y = np.linspace(a, b, M, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = np.linspace(0, N*dt, N, endpoint=False)

    psi[0,:] = psi0(X, Y)

    for i in range(1, N):
        ps = psi[i-1,:]
        psi1 = ps * np.exp(-1j*((x**2+y**2)/2+k1*np.abs(ps)**2)*dt/(2*eps))
        psihat1 = fft.fft2(psi1)
        freq = freq_maker(M)
        freqx, freqy = np.meshgrid(freq, freq, sparse=False, indexing="ij")
        psihat2 = psihat1 * np.exp(-1j* eps*dt*4*np.pi**2*(freqx**2+freqy**2)/(b-a)**2)
        psi2 = fft.ifft2(psihat2)
        psi[i] = psi2 * np.exp(-1j*((x**2+y**2)/2+k1*np.abs(psi2)**2)*dt/(2*eps))

    return t, X, Y, psi



def timeindep_tssp_2d_step():

    return Null



def timedep_gp_1d(m, time_steps, a, b, psi0, beta, dt, potential):

    grid_points = 2**m
    psi = np.empty((time_steps, grid_points), dtype=complex)

    x = np.linspace(0, 1, grid_points, endpoint=False)
    t = np.linspace(0, time_steps*dt, time_steps, endpoint=False)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)

    V = potential(x)
    expV = np.exp(-k*V)
    zero_pot = V == 0

    psi[0,:] = psi0(x)

    for i in range(N-1):
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

    x = np.linspace(a, b, M, endpoint=False)
    y = np.linspace(a, b, M, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")

    if psi.shape == x.shape:
        return np.sum(f(X,Y) * np.abs(psi)**2 * dA)
    else:
        raise RuntimeError("Size of psi should match the size of the grid.")

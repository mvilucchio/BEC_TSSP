import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

EPS = 1e-12

def get_freq(N):
    arr = np.zeros((N,))
    arr[0:int(N-np.floor(N/2))] = np.array(range(0,int(N-np.floor(N/2))))
    arr[int(N-np.floor(N/2)):N] = np.array(range(0,int(np.floor(N/2)))) - np.floor(N/2)
    return arr



def ti_tssp_1d_pbc(m, N, a, b, psi0, potential, dt, k, eps):

    grid_points = 2**m

    psi = np.empty((N, grid_points), dtype=complex)

    x = np.linspace(a, b, grid_points, endpoint=False)
    t = np.linspace(0, N*dt, N, endpoint=False)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)

    V = potential(x)

    psi[0,:] = psi0(x)

    for n in range(N-1):
        psi[n+1,:] = _ti_tssp_1d_pbc_step(psi[n,:], V)

    return t, x, psi



def _ti_tssp_1d_pbc_step(psi, V):

    p1 = psi * np.exp(-1j*(V + k * np.abs(psi)**2) * dt/(2*eps))
    p2 = fft.ifft(np.exp(-1j*eps*dt*mu**2/2) * fft.fft(p1))
    return p2 * np.exp(-1j*(V + k * np.abs(p2)**2)*dt/(2*eps))



def ti_tssp_2d_pbc(M, N, q, a, b, psi0, potential, dt, beta, eps):
    # 1/q is the fraction of the times we want to store. q must devide N

    n = int(N/q)
    x = np.linspace(a, b, M, endpoint=False)
    y = np.linspace(a, b, M, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = np.linspace(0, N*dt, n, endpoint=False)
    f = get_freq(M)
    Fx, Fy = np.meshgrid(freq,freq,sparse=False,indexing="ij")
    F2 = Fx**2 + Fy**2

    psi = np.empty((n, M, M), dtype=complex)
    psi[0,:,:] = psi0(X, Y)

    V = potential(X, Y)

    for i in range(1,n):
        psi[i,:] = _td_tssp_2d_pbc_multi_step(psi[i-1,:], beta, eps, dt, q, b-a, V, F2)

    return t, X, Y, psi


# can be made faster even more
def _ti_tssp_2d_pbc_multi_step(psi, beta, eps, dt, q, len, V, F2):

    ps = psi * np.exp(-1j*(V + beta*np.abs(psi)**2)*dt/(2*eps))
    ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * eps*dt * (4*np.pi**2/len**2) * F2))

    for j in range(q-1):
        ps = ps * np.exp(-1j*(V + beta*np.abs(ps)**2)*dt/eps)
        ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j* eps*dt * (4*np.pi**2/len**2) * F2))

    return ps * np.exp(-1j*(V + beta*np.abs(ps)**2)*dt/(2*eps))



def td_tssp_2d_pbc(M, time_steps, a, b, psi0, potential, dt, beta, eps):

    #n = int(time_steps/every_n_t)
    x = np.linspace(a, b, M, endpoint=False)
    y = np.linspace(a, b, M, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = dt * np.arange(time_steps)
    f = get_freq(M)

    Fx, Fy = np.meshgrid(f, f, sparse=False, indexing="ij")
    F2 = Fx**2 + Fy**2

    psi = np.empty((time_steps, M, M), dtype=complex)
    psi[0,:] = psi0(X, Y)

    V = potential(X, Y) / eps
    zero_pot = V == 0.0
    expV = np.exp(- dt * V / eps)

    p = psi[0,:]

    for i in range(1, time_steps):
        psi[i,:] = _td_tssp_pbc_2d_step(psi[i-1,:], dt, beta, eps, x[2]- x[1], x[2]- x[1], b-a, \
                                        V, expV, zero_pot, F2)

    return t, X, Y, psi



def _td_tssp_pbc_2d_step(psi, dt, beta, eps, dx, dy, len, V, expV, zero_pot, F2):

    abs_psi = np.abs(psi)**2
    p1 = np.empty(psi.shape)
    np.putmask(p1, ~zero_pot, psi * np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_psi)))
    np.putmask(p1, zero_pot, psi * 1 / np.sqrt(1 + beta*dt * abs_psi))
    #p1[zero_pot] = psi * 1 / np.sqrt(1 + beta*dt * abs_psi)

    p2 = fft.ifft2(fft.fft2(p1) * np.exp(-eps*dt * (4*np.pi**2/len**2) * F2))

    abs_p2 = np.abs(p2)**2
    p3 = np.empty(psi.shape)
    np.putmask(p3, ~zero_pot, p2 * np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_p2)))
    np.putmask(p3, zero_pot, p2 * 1 / np.sqrt(1 + beta*dt * abs_p2))
    #p3[zero_pot] = p2 * 1 / np.sqrt(1 + beta*dt * abs_p2)

    return p3 / np.sqrt(dx*dy * np.sum(np.abs(p3)))



def mean_value(f, psi, a, b, M):
    """
    Return the mean value of a function evaluated on a square grid of size
    (b-a)^2 with M^2 points on it w.r.t. the probability density defined by
    the wavefunction psi.

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



def gradient_2d(psi, x_spacing, y_spacing):
    g = np.empty((2, psi.shape[0], psi.shape[1]))
    g[0,:] = (np.roll(psi, 1, axis=0) - psi)/x_spacing
    g[1,:] = (np.roll(psi, 1, axis=1) - psi)/y_spacing
    return g



def energy_gpe(psi, V, beta, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = g[0,:]**2 + g[1,:]**2
    return np.sum(0.5*g2 + V * a**2 + 0.5*beta * a**4)

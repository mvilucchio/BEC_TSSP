import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

EPS = 1e-12

def get_freq(N):
    arr = np.zeros((N,))
    arr[0:int(N-np.floor(N/2))] = np.array(range(0,int(N-np.floor(N/2))))
    arr[int(N-np.floor(N/2)):N] = np.array(range(0,int(np.floor(N/2)))) - np.floor(N/2)
    return arr



def get_mu(M, leng):
    arr = np.zeros((M,))
    arr[0:int(M-np.floor(M/2))] = np.array(range(0,int(M-np.floor(M/2))))
    arr[int(M-np.floor(M/2)):M] = np.array(range(0,int(np.floor(M/2)))) - np.floor(M/2)
    return 2 * np.pi * arr/leng



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



def td_tssp_2d_pbc(M, time_steps, saving_time, x_range, y_range, y_max, psi0, potential, dt, beta, eps):

    if time_steps % saving_time != 0:
        raise ValueError('The parameter saving_time should divide time_steps.')

    if x_range is not list or y_range is not list:
        raise ValueError('The parameters x_range and y_range should be lists.')

    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError('The parameters x_range and y_range should be of two elements.')

    x_max = x_range[1]
    x_min = x_range[0]
    y_max = y_range[1]
    y_min = y_range[0]

    x = np.linspace(x_min, x_max, M, endpoint=False)
    y = np.linspace(y_min, y_max, M, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = dt * np.arange(time_steps / saving_time)
    mux = get_mu(M, x_max - x_min)
    muy = get_mu(M, y_min - y_max)

    Mux, Muy = np.meshgrid(mux, muy, sparse=False, indexing="ij")
    Mu2 = Mux**2 + Muy**2

    psi = np.empty((time_steps/saving_time, M, M), dtype=complex)
    psi[0,:] = psi0(X, Y)

    V = potential(X, Y) / eps
    zero_pot = (np.abs(V) < EPS).astype(int)
    expV = np.exp(- dt * V )

    p = psi[0,:]

    for i in range(1, time_steps):
        p = _td_tssp_pbc_2d_step(p, dt, beta/eps, eps, x[2] - x[1], y[2] - y[1], \
                                        V, expV, zero_pot, Mu2)
        if i % saving_time == 0:
            psi[i / saving_time,:] = p

    return t, X, Y, psi



def _td_tssp_pbc_2d_step(psi, dt, beta, eps, dx, dy, V, expV, zero_pot, Mu2):

    abs_psi = np.abs(psi)**2
    p1 = np.empty(psi.shape, dtype=complex)
    p1 = psi * np.where(zero_pot, 1 / np.sqrt(1 + beta*dt * abs_psi), \
                        np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_psi)) )

    p2 = fft.ifft2(fft.fft2(p1) * np.exp(- eps*dt * Mu2 /2))

    abs_p2 = np.abs(p2)**2
    p3 = np.empty(psi.shape, dtype=complex)
    p3 = p2 * np.where(zero_pot, 1 / np.sqrt(1 + beta*dt * abs_p2), \
                        np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_p2)))

    return p3 / np.sqrt(dx*dy * np.sum(np.abs(p3)**2))



def mean_value_2d(f, psi, x_range, y_range, M):

    if x_range is not list or y_range is not list:
        raise ValueError('The parameters x_range and y_range should be lists.')

    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError('The parameters x_range and y_range should be of two elements.')

    x_max = x_range[1]
    x_min = x_range[0]
    y_max = y_range[1]
    y_min = y_range[0]

    dA = (b-a)*(d-c)/(M**2)

    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")

    if psi.shape == x.shape:
        return np.sum(f(X,Y) * np.abs(psi)**2) * dA
    else:
        raise ValueError("Size of psi should match the size of the grid.")



def gradient_2d(psi, x_spacing, y_spacing):
    g = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    g[0,:] = (psi - np.roll(psi, 1, axis=0))/x_spacing
    g[1,:] = (psi - np.roll(psi, 1, axis=1))/y_spacing
    return g



def veloc_2d(psi, x_spacing, y_spacing):
    v = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    zero_abs = (np.abs(psi)**2 < -5).astype(int)
    v = np.where(zero_abs, 0, gradient_2d(psi, x_spacing, y_spacing) * np.conj(psi) - \
                 psi * gradient_2d(np.conj(psi), x_spacing, y_spacing)) / (1j*np.abs(psi)**2)
    return v




def energy_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = g[0,:]**2 + g[1,:]**2
    return x_spacing * y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + 0.5*beta/eps * a**4)



def mu_gpe(psi, V, beta, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = np.abs(g[0,:])**2 + np.abs(g[1,:])**2
    return x_spacing*y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + beta/eps * a**4)

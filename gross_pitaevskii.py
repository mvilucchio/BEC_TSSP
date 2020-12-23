import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from inspect import signature

EPS = 1e-12

def _progress_bar(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)



def get_mu(M, len):
    arr = np.zeros((M,))
    arr[0:int(M-np.floor(M/2))] = np.array(range(0,int(M-np.floor(M/2))))
    arr[int(M-np.floor(M/2)):M] = np.array(range(0,int(np.floor(M/2)))) - np.floor(M/2)
    return 2 * np.pi * arr/len



def ti_tssp_1d_pbc(m, N, a, b, psi0, potential, dt, k, eps, verbose=True):

    grid_points = 2**m

    psi = np.empty((N, grid_points), dtype=complex)

    x = np.linspace(a, b, grid_points, endpoint=False)
    t = np.linspace(0, N*dt, N, endpoint=False)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)

    V = potential(x)

    psi[0,:] = psi0(x)

    for n in range(N-1):
        psi[n+1,:] = _ti_tssp_1d_pbc_step(psi[n,:], V)

        if verbose:
            _progress_bar(percent = int(n / (N - 2) * 100))

    if verbose:
        print('')

    return t, x, psi



def _ti_tssp_1d_pbc_step(psi, V):

    p1 = psi * np.exp(-1j*(V + k * np.abs(psi)**2) * dt/(2*eps))
    p2 = fft.ifft(np.exp(-1j*eps*dt*mu**2/2) * fft.fft(p1))
    return p2 * np.exp(-1j*(V + k * np.abs(p2)**2)*dt/(2*eps))



def ti_tssp_2d_pbc(grid_points, time_steps, saving_time, x_range, y_range, psi0, potential, dt, beta, eps, verbose=True):

    if time_steps % saving_time != 0:
        raise ValueError('The parameter saving_time should divide time_steps.')

    if not isinstance(x_range, list) or not isinstance(y_range, list):
        raise ValueError('The parameters x_range and y_range should be lists.')

    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError('The parameters x_range and y_range should be lists of two elements.')

    param_potential = signature(potential).parameters

    if len(param_potential) > 3 or len(param_potential) < 2:
        raise ValueError('The function potential should have two paramethers (x, y) or three paramethers (x, y, t).')
    elif len(param_potential) == 3:
        flag_pot_time = True
    else:
        flag_pot_time = False

    if isinstance(psi0, np.ndarray):
        if psi0.shape != (grid_points, grid_points):
            raise ValueError('The size of psi0 should be (grid_points, grid_points), in this case ({},{}).'.format(grid_points, grid_points))
        flag_psi0_ndarray = True
    elif callable(psi0):
        param_psi0 = signature(psi0).parameters
        if len(param_psi0) != 2:
            raise ValueError('The funcion psi0 should have two paramethers (x, y).')
        flag_psi0_ndarray = False
    else:
        raise TypeError('The paramether psi0 should be either a function or a numpy.ndarray instance.')

    x_max = x_range[1]
    x_min = x_range[0]
    y_max = y_range[1]
    y_min = y_range[0]

    n = time_steps // saving_time
    x = np.linspace(x_min, x_max, grid_points, endpoint=False)
    y = np.linspace(y_min, y_max, grid_points, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = np.linspace(0, time_steps*dt, n, endpoint=False)

    mux = get_mu(grid_points, x_max - x_min)
    muy = get_mu(grid_points, y_min - y_max)

    Mux, Muy = np.meshgrid(mux, muy, sparse=False, indexing="ij")
    Mu2 = Mux**2 + Muy**2

    psi = np.empty((n, grid_points, grid_points), dtype=complex)
    if flag_psi0_ndarray:
        psi[0,:,:] = psi0
    else:
        psi[0,:,:] = psi0(X, Y)

    if flag_pot_time:
        for i in range(1,n):
            psi[i,:] = _ti_tssp_2d_pbc_multi_step_time(psi[i-1,:], beta, eps, dt, \
                                                  saving_time, potential, X, Y, t[i], Mu2)

            if verbose:
                _progress_bar(percent = int(i / (n - 1) * 100))

    else:
        V = potential(X, Y)

        for i in range(1,n):
            psi[i,:] = _ti_tssp_2d_pbc_multi_step(psi[i-1,:], beta, eps, dt, \
                                                  saving_time, V, Mu2)

            if verbose:
                _progress_bar(percent = int(i / (n - 1) * 100))

    if verbose:
        print('')

    return t, X, Y, psi



def _ti_tssp_2d_pbc_multi_step(psi, beta, eps, dt, saving_time, V, Mu2):

    ps = psi * np.exp(-1j*(V + beta*np.abs(psi)**2)*dt/(2*eps))
    ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    for j in range(saving_time-1):
        ps = ps * np.exp(-1j*(V + beta*np.abs(ps)**2)*dt/eps)
        ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    return ps * np.exp(-1j*(V + beta*np.abs(ps)**2)*dt/(2*eps))



def _ti_tssp_2d_pbc_multi_step_time(psi, beta, eps, dt, saving_time, potential, X, Y, t, Mu2):

    ps = psi * np.exp(-1j*(potential(X, Y, t) + beta*np.abs(psi)**2)*dt/(2*eps))
    ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    for j in range(saving_time-1):
        ps = ps * np.exp(-1j*(potential(X, Y, t + dt*j) + beta*np.abs(ps)**2)*dt/eps)
        ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    return ps * np.exp(-1j*(potential(X, Y, t + dt*j) + beta*np.abs(ps)**2)*dt/(2*eps))



def td_tssp_2d_pbc(grid_points, time_steps, saving_time, x_range, y_range, psi0, potential, dt, beta, eps, verbose=True):

    if time_steps % saving_time != 0:
        raise ValueError('The parameter saving_time should divide time_steps.')

    if not isinstance(x_range, list) or not isinstance(y_range, list):
        raise ValueError('The parameters x_range and y_range should be lists.')

    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError('The parameters x_range and y_range should be lists of two elements.')

    x_max = x_range[1]
    x_min = x_range[0]
    y_max = y_range[1]
    y_min = y_range[0]

    n = time_steps // saving_time
    x = np.linspace(x_min, x_max, grid_points, endpoint=False)
    y = np.linspace(y_min, y_max, grid_points, endpoint=False)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")
    t = dt * np.arange(time_steps / saving_time)
    mux = get_mu(grid_points, x_max - x_min)
    muy = get_mu(grid_points, y_min - y_max)

    Mux, Muy = np.meshgrid(mux, muy, sparse=False, indexing="ij")
    Mu2 = Mux**2 + Muy**2

    psi = np.empty((n, grid_points, grid_points), dtype=complex)
    psi[0,:] = psi0(X, Y)

    V = potential(X, Y) / eps
    zero_pot = (np.abs(V) < EPS).astype(int)
    expV = np.exp(- dt * V )

    p = psi[0,:]


    #old_dict = np.seterr(divide='ignore', invalid='ignore')

    for i in range(1, time_steps):
        p = _td_tssp_pbc_2d_step(p, dt, beta/eps, eps, x[2] - x[1], y[2] - y[1], \
                                        V, expV, zero_pot, Mu2)
        if verbose:
            _progress_bar(percent = int(i / (time_steps - 1) * 100))

        if i % saving_time == 0:
            psi[i // saving_time,:] = p

    if verbose:
        print('')

    #_ = np.seterr(old_dict)

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

    if not isinstance(x_range, list) or not isinstance(y_range, list):
        raise ValueError('The parameters x_range and y_range should be lists.')

    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError('The parameters x_range and y_range should be of two elements.')

    x_max = x_range[1]
    x_min = x_range[0]
    y_max = y_range[1]
    y_min = y_range[0]

    dA = (x_max - x_min)*(y_max - y_min)/(M**2)

    x = np.linspace(x_min, x_max, M)
    y = np.linspace(y_min, y_max, M)
    X, Y = np.meshgrid(x, y, sparse=False, indexing="ij")

    if psi.shape == X.shape:
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



def mu_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = np.abs(g[0,:])**2 + np.abs(g[1,:])**2
    return x_spacing*y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + beta/eps * a**4)

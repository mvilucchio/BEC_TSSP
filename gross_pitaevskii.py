import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from inspect import signature

EPS = 1e-12
PRINT_EACH = 100


def _progress_bar(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.0f}%', sep='', end='', flush=True)


def get_mu(M, len):
    arr = np.zeros((M,))
    arr[0:int(M-np.floor(M/2))] = np.array(range(0, int(M-np.floor(M/2))))
    arr[int(M-np.floor(M/2)):M] = np.array(range(0, int(np.floor(M/2)))) - np.floor(M/2)
    return 2 * np.pi * arr/len


def ti_tssp_1d_pbc(grid_points, time_steps, saving_time, x_range, psi0, potential, dt, beta, eps, verbose=True):

    if time_steps % saving_time != 0:
        raise ValueError('The parameter saving_time should divide time_steps.')

    if not isinstance(x_range, list):
        raise ValueError('The parameter x_range should be a list.')

    if len(x_range) != 2:
        raise ValueError('The parameter x_range should be list a of two elements.')

    n = int(time_steps / saving_time)

    x_max = x_range[1]
    x_min = x_range[0]

    psi = np.empty((n, grid_points), dtype=complex)

    x = np.linspace(x_min, x_max, grid_points, endpoint=False)
    t = np.linspace(0, time_steps*dt, n, endpoint=False)
    mu = 2*np.pi * np.arange(grid_points, dtype=float)
    mu2 = mu**2

    V = potential(x)

    psi[0, :] = psi0(x)

    for i in range(1, n):
        psi[i, :] = _ti_tssp_1d_pbc_multi_step(psi[i-1, :], beta, eps, dt, saving_time, V, mu2)

        if verbose and (i % PRINT_EACH == 0 or i == n-1):
            _progress_bar(percent=int(i / (n - 2) * 100))

    if verbose:
        print('')

    return t, x, psi


def _ti_tssp_1d_pbc_multi_step(psi, beta, eps, dt, saving_time, V, mu2):

    ps = psi * np.exp(-1j * (V + beta * np.abs(psi)**2) * dt/(2*eps))
    ps = fft.ifft(np.exp(-1j * 0.5*eps*dt * mu2) * fft.fft(ps))

    for j in range(saving_time-1):
        ps = psi * np.exp(-1j * (V + beta * np.abs(psi)**2) * dt/(2*eps))
        ps = fft.ifft(np.exp(-1j * 0.5*eps*dt * mu2) * fft.fft(ps))

    return ps * np.exp(-1j * (V + beta * np.abs(ps)**2) * dt/(2*eps))


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
            raise ValueError('The size of psi0 should be (grid_points, grid_points), in this case ({},{}).'.format(
                grid_points, grid_points))
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
        psi[0, :, :] = psi0
    else:
        psi[0, :, :] = psi0(X, Y)

    if flag_pot_time:
        for i in range(1, n):
            psi[i, :] = _ti_tssp_2d_pbc_multi_step_time(psi[i-1, :], beta, eps, dt,
                                                        saving_time, potential, X, Y, t[i], Mu2)

            if verbose:
                _progress_bar(percent=int(i / (n - 1) * 100))

    else:
        V = potential(X, Y)

        for i in range(1, n):
            psi[i, :] = _ti_tssp_2d_pbc_multi_step(psi[i-1, :], beta, eps, dt,
                                                   saving_time, V, Mu2)

            if verbose:
                _progress_bar(percent=int(i / (n - 1) * 100))

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
    psi[0, :] = psi0(X, Y)

    V = potential(X, Y) / eps
    zero_pot = (np.abs(V) < EPS).astype(int)
    expV = np.exp(- dt * V)

    p = psi[0, :]

    # done to prevent numpy Warning taken care with the where function
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(1, time_steps):
            p = _td_tssp_pbc_2d_step(p, dt, beta/eps, eps,
                                     x[2] - x[1], y[2] - y[1], V, expV, zero_pot, Mu2)

            if verbose and (i % PRINT_EACH == 0 or i == time_steps-1):
                _progress_bar(percent=int(i / (time_steps - 1) * 100))

            if i % saving_time == 0:
                psi[i // saving_time, :] = p

    if verbose:
        print('')

    return t, X, Y, psi


def _td_tssp_pbc_2d_step(psi, dt, beta, eps, dx, dy, V, expV, zero_pot, Mu2):

    abs_psi = np.abs(psi)**2
    p1 = np.empty(psi.shape, dtype=complex)
    p1 = psi * np.where(zero_pot, 1 / np.sqrt(1 + beta*dt * abs_psi),
                        np.sqrt((V*expV) / (V + beta*(1 - expV)*abs_psi)))

    p2 = fft.ifft2(fft.fft2(p1) * np.exp(- eps*dt * Mu2 / 2))

    abs_p2 = np.abs(p2)**2
    p3 = np.empty(psi.shape, dtype=complex)
    p3 = p2 * np.where(zero_pot, 1 / np.sqrt(1 + beta*dt * abs_p2),
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
        return np.sum(f(X, Y) * np.abs(psi)**2) * dA
    else:
        raise ValueError("Size of psi should match the size of the grid.")


def gradient_2d(psi, x_spacing, y_spacing):
    g = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    g[0, :] = (psi - np.roll(psi, 1, axis=0))/x_spacing
    g[1, :] = (psi - np.roll(psi, 1, axis=1))/y_spacing
    return g


def veloc_2d(psi, x_spacing, y_spacing):
    v = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    zero_abs = (np.abs(psi)**2 < -5).astype(int)
    v = np.where(zero_abs, 0, gradient_2d(psi, x_spacing, y_spacing) * np.conj(psi) -
                 psi * gradient_2d(np.conj(psi), x_spacing, y_spacing)) / (1j*np.abs(psi)**2)
    return v


def energy_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = g[0, :]**2 + g[1, :]**2
    return x_spacing * y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + 0.5*beta/eps * a**4)


def mu_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = np.abs(g[0, :])**2 + np.abs(g[1, :])**2
    return x_spacing*y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + beta/eps * a**4)


def _crop_array_idxs(array, val_min, val_max):
    """
    Given an increaslingly ordered array and two values (val_min and val_max) it
    returns the indicies of the smallest element still greater or equal to val_min
    and the greater element still smaller or equal to val_max.

    Parameters
    ----------
    array : list or numpy.ndarray
        Sorted array.
    val_min : float
        Minimal value of the elements.
    val_max : float
        Maximal value of the elements.

    Returns
    -------
    idx_min, idx_max
        Indicies within which there are elements smaller than val_max and bigger
        then val_min.

    """

    flag_min = True
    flag_max = True
    l = len(array)

    for i, elem in enumerate(array):
        if flag_min:
            if elem >= val_min:
                idx_min = i
                flag_min = False
        if flag_max:
            if array[l - 1 - i] <= val_max:
                idx_max = l - 1 - i
                flag_max = False
        if not flag_max and not flag_min:
            break

    if flag_min or flag_max:
        raise RuntimeError('Indeces not found.')

    return idx_min, idx_max


def winding_number(X, Y, Z, contour, x_spacing, y_spacing):
    """
    Short summary.

      A (x_1, y_2) +---<---+ D (x_2, y_2)
                   |       |
                   v       ^
                   |       |
      B (x_1, y_1) +--->---+ C (x_2, y_1)

    Parameters
    ----------
    X : numpy.ndarray
        Matrix with values for the first axis on all the rows.
    Y : numpy.ndarray
        Matrix with values for the second axis on all the columns.
    Z : numpy.ndarray
        Either a matrix with 3 dimension and the last two dimensions like the
        dimensions of X and Y or a list of two matricies with the same size as
        X and Y.
    contour : list of floats
        Should contain the extrems of the path.

    Returns
    -------
    float
        Value of the contour integreation around the contour.

    """

    if isinstance(Z, list):
        if len(Z) != 2:
            raise ValueError("The argument z should be a list of two elements.")
        else:
            q_x = Z[0]
            q_y = Z[1]
    elif isinstance(Z, np.ndarray):
        if len(Z.shape) != 3 or Z.shape[0] < 2:
            raise ValueError(
                "The argument z should be a numpy array of dimension 3 with at least 2 values on the first axis.")
        else:
            q_x = Z[0, :]
            q_y = Z[1, :]
    else:
        raise TypeError(
            "The argument z should be a list of numpy.ndarray or an instance of numpy.ndarray.")

    if not isinstance(contour, list):
        raise TypeError('The argument contour should be a list of tuples.')
    elif len(contour) != 4:
        raise ValueError(
            'The number of elements in contour should be 4, here it is {}'.format(len(contour)))

    x_1, x_2, y_1, y_2 = contour

    if x_2 < x_1 or y_2 < y_1:
        raise ValueError('It should be x_1 < x_2 and y_1 < y_2.')

    x_1_idx, x_2_idx = _crop_array_idxs(X[:, 0], x_1, x_2)
    y_1_idx, y_2_idx = _crop_array_idxs(Y[0, :], y_1, y_2)

    int_value = 0.0
    int_value += q_x[x_1_idx: x_2_idx, y_1_idx].sum() * x_spacing
    int_value += q_y[x_2_idx, y_1_idx: y_2_idx].sum() * y_spacing
    int_value -= q_x[x_1_idx: x_2_idx, y_2_idx].sum() * x_spacing
    int_value -= q_y[x_2_idx, y_1_idx: y_2_idx].sum() * y_spacing

    return int_value


def phase_slip(psi_t):
    return None

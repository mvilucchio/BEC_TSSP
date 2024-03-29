import numpy as np
from numpy import fft
from inspect import signature

EPS = 1e-12
PRINT_EACH = 100

def _progress_bar(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.0f}%', sep='', end='', flush=True)


def get_mu(M, l):
    """
    Helper function to calucate the value of mu used to in the TSSP method.

    Parameters
    ----------
    M : int
        [description]
    l : int
        [description]

    Returns
    -------
    float
        Value of mu.
    """
    arr = np.zeros((M,))
    arr[0:int(M-np.floor(M/2))] = np.array(range(0, int(M-np.floor(M/2))))
    arr[int(M-np.floor(M/2)):M] = np.array(range(0, int(np.floor(M/2)))) - np.floor(M/2)
    return 2 * np.pi * arr/l


def ti_tssp_1d_pbc(grid_points, time_steps, saving_time, x_range, psi0, potential, dt, beta, eps, verbose=True):
    """
    Evolution for the 1D Gross Pitaevskii equation solve with TSSP.

    Parameters
    ----------
    grid_points : [type]
        [description]
    time_steps : int
        Total number of time steps to simulate evolution.
    saving_time : int
        
    x_range : list of floats
        List containing the minimum and maximum value of the simulation line.
    psi0 : numpy.ndarray
        Initial value of the wave functoin
    potential : function
        External potential to which the wavefunction si subject to.
    dt : float
        [description]
    beta : [type]
        [description]
    eps : [type]
        [description]
    verbose : bool, optional
        [description], by default True.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        The parameter `saving_time` should divide `time_steps`.
    ValueError
        The parameter `x_range` should be a list.
    ValueError
        The parameter `x_range` should be list a of two elements.
    """
    if time_steps % saving_time != 0:
        raise ValueError('The parameter `saving_time` should divide `time_steps`.')

    if not isinstance(x_range, list):
        raise ValueError('The parameter `x_range` should be a list.')

    if len(x_range) != 2:
        raise ValueError('The parameter `x_range` should be list a of two elements.')

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

    n = int(time_steps / saving_time)
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

    ps = psi * np.exp(-1j*((V + beta*np.abs(psi)**2)*dt)/(2*eps))
    ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    for j in range(saving_time-1):
        ps = ps * np.exp(-1j*((V + beta*np.abs(ps)**2)*dt)/eps)
        ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    return ps * np.exp(-1j*((V + beta*np.abs(ps)**2)*dt)/(2*eps))


def _ti_tssp_2d_pbc_multi_step_time(psi, beta, eps, dt, saving_time, potential, X, Y, t, Mu2):

    ps = psi * np.exp(-1j*(potential(X, Y, t) + beta*np.abs(psi)**2)*dt/(2*eps))
    ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    for j in range(saving_time-1):
        ps = ps * np.exp(-1j*(potential(X, Y, t + dt*j) + beta*np.abs(ps)**2)*dt/eps)
        ps = fft.ifft2(fft.fft2(ps) * np.exp(-1j * 0.5*eps*dt * Mu2))

    return ps * np.exp(-1j*(potential(X, Y, t + dt*j) + beta*np.abs(ps)**2)*dt/(2*eps))


def td_tssp_2d_pbc(grid_points, time_steps, saving_time, x_range, y_range, psi0, potential, dt, beta, eps, verbose=True):
    """
    Generates the time evolution of the Time Dependant (TD) Gross Pitaevski equation solved with Periodic Boundary Conditions in 2D.

    Parameters
    ----------
    grid_points : [type]
        [description]
    time_steps : [type]
        [description]
    saving_time : [type]
        [description]
    x_range : [type]
        [description]
    y_range : [type]
        [description]
    psi0 : [type]
        [description]
    potential : [type]
        [description]
    dt : [type]
        [description]
    beta : [type]
        [description]
    eps : [type]
        [description]
    verbose : bool, optional
        [description], by default True

    Returns
    -------
    (t, x, y, psi)
        t : numpy.ndarray
            Times where the evolution has been recorded in array `psi`
        X : numpy.ndarray
            Mesh grid corresponding to the first axis.
        Y : numpy.ndarray
            Mesh grid corresponding to the second axis.
        psi : numpy.ndarray
            Values of the wavefunction for the recorded timesteps. The first dimension has the same number of elements as `t` and it contains all the time steps. The other two correspond to the two spatial dimensions.

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    """
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
    """
    Evaluation of the mean value of the function in the given interval.

    Parameters
    ----------
    f : function
        Function that takes two matrices X and Y and is mutiplied to psi before taking the mean.
    psi : numpy.ndarray
        Values of the wavefunction.
    x_range : list of floats
        List containing as first/second element the minimum/maximum value of the x axis.
    y_range : list of floats
        List containing as first/second element the minimum/maximum value of the y axis.
    M : int
        Number of steps in each axis.

    Raises
    ------
    ValueError
        It is returned if the size of psi is not the same of the as the points in the grid.
        Returned if the arguments `x_range` and `y_range` are not lists.
        Returned if the legnths of `x_range` and `y_range` are not of 2.

    Returns
    -------
    numpy.ndarray
        Description of returned object.

    """

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
    """
    Evaluates the discrete gradient of the function psi on the grid given by x_spacing and y_spacing.

    Parameters
    ----------
    psi : numpy.ndarray
        Function to calculate the gradient from.
    x_spacing : float
        Discretization spacing on the x axis.
    y_spacing : float
        Discretization spacing on the x axis.

    Returns
    -------
    numpy.ndarray
        Array containing the gradient of psi.

    """
    g = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    g[0, :] = (psi - np.roll(psi, 1, axis=0))/x_spacing
    g[1, :] = (psi - np.roll(psi, 1, axis=1))/y_spacing
    return g


def veloc_2d(psi, x_spacing, y_spacing):
    """
    Calculates the velocity field of a given discretized wavefunction `psi` and discretization on the two axis `x_spacing` and `y_spacing`.

    Parameters
    ----------
    psi : numpy.ndarray
        Values of the wavefunction for every grid step.
    x_spacing : float
        Length of the spacing along the first axis.
    y_spacing : float
        Legnth of the spacing along the second axis.

    Returns
    -------
    type
        Description of returned object.

    """
    v = np.empty((2, psi.shape[0], psi.shape[1]), dtype=psi.dtype)
    zero_abs = (np.abs(psi)**2 < -5).astype(int)
    v = np.where(zero_abs, 0, gradient_2d(psi, x_spacing, y_spacing) * np.conj(psi) -
                 psi * gradient_2d(np.conj(psi), x_spacing, y_spacing)) / (1j*np.abs(psi)**2)
    return v


def energy_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    """
    Given the paramether of the Gross-Pitaevskii equation it computes the
    energy for the given wavefunction.

    Parameters
    ----------
    psi : numpy.ndarray
        Numpy matrix with the values of the wavefunction on the grid. This should
        have the same size as V.
    V : numpy.ndarray
        Values of the potential on the same grid as psi. This should have the same
        size as psi.
    beta : float
        Non linearity paramether of the Gross-Pitaevskii equation.
    eps : float
        Value of the time separation step.
    x_spacing : float
        Distance between two points in the x axis.
    y_spacing : float
        Distance between two points in the y axis.

    Returns
    -------
    type
        Computed value of the energy.

    """
    a = np.abs(psi)
    g = gradient_2d(psi, x_spacing, y_spacing)
    g2 = g[0, :]**2 + g[1, :]**2
    return x_spacing * y_spacing * np.sum(0.5*eps * g2 + V/eps * a**2 + 0.5*beta/eps * a**4)


def mu_gpe(psi, V, beta, eps, x_spacing, y_spacing):
    """
    Given the paramether of the Gross-Pitaevskii equation it computes the
    chemical potential for the given wavefunction.

    Parameters
    ----------
    psi : numpy.ndarray
        Numpy matrix with the values of the wavefunction on the grid. This should
        have the same size as V.
    V : numpy.ndarray
        Values of the potential on the same grid as psi. This should have the same
        size as psi.
    beta : float
        Non linearity paramether of the Gross-Pitaevskii equation.
    eps : float
        Value of the time separation step.
    x_spacing : float
        Distance between two points in the x axis.
    y_spacing : float
        Distance between two points in the y axis.

    Returns
    -------
    float
        Computed value of the chemical potential.

    """
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
    This method computes the contour integreation around a positively oriented
    square of corners A, B, C and D.

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

    Raises
    ------
    ValueError
        If the shape of z or contour is not a valid list.
    TypeError
        If the type of any of the initial agruments don't match the required.

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


def calc_phase(psi):

    phases = np.empty(len(psi))
    #zero_abs = (np.abs(ps)**2 < -10**(-2)).astype(int)
    phases = np.arctan2(psi.imag, psi.real)
    return phases

#Here I only calculate the phase for the values of y greatter than zero, which are in fact the relevant ones
def delta_S(psi_time, M):

    phases = np.zeros((len(psi_time), int(M/2)))
    dif_phase = np.zeros((len(psi_time), int(M/2)))

    for i, psi in enumerate(psi_time):
        phases[i] = calc_phase( psi[int(M/2)][int(M/2):] )
        dif_phase[i] = np.angle( np.exp(1j * np.roll(phases[i],1)) * np.exp(-1j * phases[i]) )

    return phases, dif_phase


def _find_indicies(array, value):
    """
    Finds the indicies of the array elements that are the closest to `value`. If
    value is present in `array` then the two indicies are the same.

    Parameters
    ----------
    value : scalar
        Value of which the index is to be found.
    array : list/numpy.ndarray
        Sorted array.

    Raises
    ------
    ValueError
        Raised if the indices are not found (i.e. the eindicies are out of bound
        of the array)

    Returns
    -------
    int, int
        Returns the indicies of the elements of the array most close to value.
        If the vlaue is present in the array then the function returns the same
        int twice.

    """

    for i, elem in enumerate(array):
        if value == elem:
            return i, i
        elif elem < value and value < array[i+1]:
            return i, i+1

    raise ValueError("Indicies not found.")



def delta_S_2(psi_time, points, X, Y, ret_max=True):
    """
    Short summary.

    Parameters
    ----------
    psi_time : numpy.ndarray
        Description of parameter `psi_time`.
    points : list of 2-tuples
        List 
    X : numpy.ndarray
        Mesh grid associated to the first axis.
    Y : numpy.ndarray
        Mesh grid associated to the second axis.
    ret_max : bool
        Description of parameter `ret_max`. The default is True.

    Raises
    ------
    TypeError
        Input `points` should be a list of tuples.
    TypeError
        Each tuple in `points` should match with (x_point, y_point) with x_point and y_point floats.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, float (optional)
        Description of returned object.

    """

    if not all([isinstance(point, tuple) for point in points]):
        raise TypeError("points should be a list of tuples.")
    elif not all([len(point) == 2 for point in points]):
        raise TypeError("Each tuple in `points` should have exactly two elements (x_point, y_point).")

    if points[0][0] == points[1][0]:
        vertical_line = True
        x = points[0][0]
        [y1, y2] = [points[0][1], points[1][1]] if points[0][1] <= points[1][1] else [points[1][1], points[0][1]]
    elif points[0][1] == points[1][1]:
        vertical_line = False
        y = points[0][1]
        [x1, x2] = [points[0][0], points[1][0]] if points[0][0] <= points[1][0] else [points[1][0], points[0][0]]
    else:
        raise ValueError("The two tuples in `points` should be [(x, y1), (x,y2)] or [(x1,y),(x2,y)].")

    if vertical_line:
        x_ind_1, x_ind_2 = _find_indicies(X[:,0], x)
        y_ind_1, y_ind_2 = _crop_array_idxs(Y[0,:], y1, y2)
        exact_match = x_ind_2 == x_ind_1
    else:
        x_ind_1, x_ind_2 = _crop_array_idxs(X[:,0], x1, x2)
        y_ind_1, y_ind_2 =_find_indicies(Y[0,:], y)
        exact_match = y_ind_2 == y_ind_2

    if vertical_line:
        phases = np.zeros((len(psi_time), y_ind_2 - y_ind_1 + 1))
        dif_phase = np.zeros((len(psi_time), y_ind_2 - y_ind_1 + 1))
    else:
        phases = np.zeros((len(psi_time), x_ind_2 - x_ind_1 + 1))
        dif_phase = np.zeros((len(psi_time), x_ind_2 - x_ind_1 + 1))

    # the choice is to always take the smallest index if ?_ind_1 and ?_ind_2 don't match
    for t, psi in enumerate(psi_time):
        if vertical_line:
            phases[t,:] = np.arctan2(psi.imag[x_ind_1, y_ind_1 : y_ind_2 + 1], psi.real[x_ind_1, y_ind_1 : y_ind_2 + 1])
        else:
            phases[t,:] = np.arctan2(psi.imag[x_ind_1 : x_ind_2 + 1, y_ind_1], psi.real[x_ind_1 : x_ind_2 + 1, y_ind_1])

        dif_phase[t,:] = np.angle( np.exp(1j * np.roll(phases[t],1)) * np.exp(-1j * phases[t]) )

    if ret_max:
        return phases, dif_phase, np.max(dif_phase)
    else:
        return phases, dif_phase

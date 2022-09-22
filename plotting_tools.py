import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

# sizes
fig_size = (6, 6)

# colourmaps
cm_surface = cm.get_cmap('coolwarm')
cm_pcolor = cm.get_cmap('Spectral')
cm_contour = cm.get_cmap('winter')

# colours
blk_background = '#121212'
text_color = '#D0D0D0'
grid_line_color = '#646464'


# remember the squeeze attribute for subplots
def _darkizer(fig, axes, title):
    """
    This function changes the theme of a figure (with axes) to a dark theme.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The object Figure of which you want to change colour.
    axes : matplotlib.axes.Axes
        The axes of the fig.
    title :
        The title of the fig.

    """

    fig.patch.set_facecolor(blk_background)

    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            axes = np.expand_dims(axes, 0)
    else:
        axes = np.array([[axes]])

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_facecolor(blk_background)

            axes[i, j].xaxis.label.set_color(text_color)
            axes[i, j].yaxis.label.set_color(text_color)

            axes[i, j].title.set_color(text_color)

            for _, spine in axes[i, j].spines.items():
                spine.set_color(text_color)

            for xline in axes[i, j].get_xgridlines():
                xline.set_color(grid_line_color)

            for yline in axes[i, j].get_ygridlines():
                yline.set_color(grid_line_color)

            for xticklab in axes[i, j].get_xticklabels():
                xticklab.set_color(text_color)

            for yticklab in axes[i, j].get_yticklabels():
                yticklab.set_color(text_color)

            for xtick in axes[i, j].get_xticklines():
                xtick.set_color(text_color)

            for ytick in axes[i, j].get_yticklines():
                ytick.set_color(text_color)


def printable(fig, axes, title, medium_size=14, big_size=16):
    """
    Function to change the font size of the text in the plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The object Figure of which you want to change colour.
    axes : matplotlib.axes.Axes
        The axes of the fig.
    title :
        The title of the fig.
    medium_size : int
        Fontsize of the medium text in the plot. The default is 14.
    big_size : int
        Fontsize of the big text in the plot. The default velue is 16.
    """

    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            axes = np.expand_dims(axes, 0)
    else:
        axes = np.array([[axes]])

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):

            for tick in axes[i, j].xaxis.get_major_ticks():
                tick.label.set_fontsize(medium_size)

            for tick in axes[i, j].yaxis.get_major_ticks():
                tick.label.set_fontsize(medium_size)

            axes[i, j].xaxis.label.set_fontsize(medium_size)
            axes[i, j].yaxis.label.set_fontsize(medium_size)
            axes[i, j].title.set_fontsize(big_size)

            lgnd = axes[i, j].get_legend()
            if lgnd != None:
                lgnd.fontsize = medium_size



def plane_plotter(x_list, y_list, title='', x_label=r'$x$', y_label=r'$y$', log_x=False, log_y=False, show_plot=True, dark=False):
    """
    Generates a simple plot of the pairs of array x and y in a plane. x_list and y_list
    can be one of the following cases:
    - x_list and y_list are both single numpy arrays and this single set of data
      will be plotted
    - x_list is a single numpy array and y_list contains multiple numpy arrays
      with the same dimension as x_list. In this case the the y_list sets of data
      will be plotted with the same horizontal coordinates.
    - x_list and y_list both correspond to multiple numpy arrays and all pari of
      data will be plotted.

    Parameters
    ----------
    x_list : numpy.ndarray / list of numpy.ndarray
        Values on the horizontal axis.
    y_list : numpy.ndarray / list of numpy.ndarray
        Values on the vertical axis.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        Name of the horizontal axis. The default is r'$x$'.
    y_label : TYPE, optional
        Name of the vertical axis. The default is r'$y$'.
    log_x : bool, optional
        The horizontal axis should be in log scale. The default is False.
    log_y : bool, optional
        The vertical axis should be in log scale. The default is False.
    show_plot: bool, optional
        Flag for printing the figure with plt.show(). The default value is True.
    dark: bool, optional
        Flag for changing the graph color to a dark theme. The default value is False.

    Raises
    ------
    ValueError
        If the size of the given list don't match any of the above mentioned cases.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """

    if not isinstance(y_list, list):
        y_list = [y_list]

    if not isinstance(x_list, list):
        x_list = [x_list]

    if len(x_list) == 1:
        x_list = x_list * len(y_list)

    if len(x_list) != len(y_list):
        raise ValueError(
            'The number of elements for x_list and y_list should be the same or 1 for x_list.')

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    for x, y in zip(x_list, y_list):
        ax.plot(x, y)

    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if show_plot:
        plt.show()

    return fig


def pcolor_plotter(X, Y, Z, title='', x_label=r'$x$', y_label=r'$y$', show_plot=True, color_bar=True, dark=False):
    """
    Generates a plane figure where the third varaible is represented as shades
    of color.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix with values for the first axis on all the rows.
    Y : numpy.ndarray
        Matrix with values for the second axis on all the columns.
    Z : numpy.ndarray
        Matrix with values of the third axis.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        Name of the horizontal axis. The default is r'$x$'.
    y_label : str, optional
        Name of the vertical axis. The default is r'$y$'.
    show_plot: bool, optional
        Flag for printing the figure with plt.show(). The default value is True.
    dark: bool, optional
        Flag for changing the graph color to a dark theme. The default value is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    pc = ax.pcolor(X, Y, Z, cmap=cm_pcolor)
    if color_bar:
        fig.colorbar(pc, shrink=0.5, aspect=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

    if show_plot:
        plt.show()

    return fig


def contour_plotter(X, Y, Z, levels=25, title='', x_label=r'$x$', y_label=r'$y$', values=False, show_plot=True, dark=False):
    """
    Generates a contour plot.

    Parameters
    ----------
    X : numpy.ndarray
        x axis mesh grid for the plot.
    Y : numpy.ndarray
        y axis mesh grid for the plot.
    Z : numpy.ndarray
        Values of the plot.
    levels : int or list of int, optional
        `levels` param in function matplotlib.pyplot.contour. The default is 25.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        Label of the horizontal axis. The default is r'$x$'.
    y_label : str, optional
        Label of the vertical axis. The default is r'$y$'.
    values : bool, optional
        Flag for having the plot contain the value of the contour lines. The
        default is False.
    show_plot : bool, optional
        if True it will call plt.show() before exiting the function. The default is True.
    dark : bool, optional
        Apply dark theme to the image. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    cp = ax.contour(X, Y, Z, levels, cmap=cm_contour)
    if values:
        ax.clabel(cp, inline=1, fontsize=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

    if show_plot:
        plt.show()

    return fig


def surface_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$', z_label=r'$z$', show_plot=True, color_bar=True, dark=False):
    """
    Generates a 3D plot.

    Parameters
    ----------
    x : numpy.ndarray
        Matrix with values for the first axis on all the rows.
    y : numpy.ndarray
        Matrix with values for the second axis on all the columns.
    z : numpy.ndarray
        Matrix with all the height of each point.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        Name of the first axis. The default is r'$x$'.
    y_label : str, optional
        Name of the second axis. The default is r'$y$'.
    z_label : str, optional
        Name of the third axis. The default is r'$z$'.
    show_plot: bool, optional
        Flag for printing the figure with plt.show(). The default value is True.
    dark : bool, optional
        Flag for changing the graph color to a dark theme. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')

    s = ax.plot_surface(x, y, z, cmap=cm_surface, linewidth=0, antialiased=False)
    if color_bar:
        fig.colorbar(s, shrink=0.5, aspect=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label, rotation=90)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

    if show_plot:
        plt.show()

    return fig


def _update_surface(frame_number, X, Y, Z, ax, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, Z[frame_number, :, :], cmap=cm_surface,
                              linewidth=0, antialiased=False)


def surface_animate(X, Y, Z, delay=200, title='', x_label=r'$x$', y_label=r'$y$', z_label=r'$z$', show_plot=True):
    """
    Animates a sequence of 3D data.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix with values for the first axis on all the rows. The size should
        be the same as Y.
    Y : numpy.ndarry
        Matrix with values for the second axis on all the columns. The size should
        be the same as X.
    Z : numpy.ndarray
        Tensor containing the height of each point and each time. The first
        dimension is for the time step and Z[i,:] should have the same dimension
        as X and Y.
    delay : int
        Time delay in milliseconds between one frame and the next one.
        The default is 200.
    title : str
        Title of the plot. The default is ''.
    x_label : str
        Name of the first axis. The default is r'$x$'.
    y_label : str
        Name of the second axis. The default is r'$y$'.
    z_label : str
        Name of the third axis. The default is r'$z$'.
    show_plot: bool, optional
        Flag for printing the figure with plt.show(). The default value is True.

    Returns
    -------
    matplotlib.animation.ArtistAnimation
        The animation of the data with the given delay.

    """

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label, rotation=90)
    ax.set_title(title)

    ax.set_zlim(Z.min(), Z.max())

    plot = [ax.plot_surface(X, Y, Z[0, :, :], cmap=cm_surface, linewidth=0, antialiased=False)]
    anim = animation.FuncAnimation(fig, _update_surface, Z.shape[0], fargs=(
        X, Y, Z, ax, plot), interval=delay, repeat_delay=2*delay, repeat=True, blit=False)

    if show_plot:
        plt.show()

    return anim


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


def quiver_plotter(X, Y, Z, plot_range=None, mes_unit='', title='', x_label=r'$x$', y_label=r'$y$', show_plot=True, dark=False):
    """
    Generates a plot of some vector fields.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix with values for the first axis on all the rows.
    Y : numpy.ndarray
        Matrix with values for the second axis on all the columns.
    Z : numpy.ndarray or list of numpy.ndarray
        Either a matrix with 3 dimension and the last two dimensions like the
        dimensions of X and Y or a list of two matricies with the same size as
        X and Y.
    plot_range : list of floats, optional
        List with the range for the plot. The defualt is None.
    mes_unit : str, optional
        Units of measure of the vectors shown. The default is ''.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        The name on the first axis. The default is r'$x$'.
    y_label : str, optional
        Name on the second axis. The default is r'$y$'.
    show_plot : bool, optional
        Flag for printing the figure with plt.show(). The default is True.
    dark : bool, optional
        Flag for changing the graph color to a dark theme. The default is False.

    Raises
    ------
    ValueError
        If the size of either X, Y or Z don't match.
    TypeError
        If the Z parameter is neither a list of numpy.ndarray or a numpy.ndarray

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

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

    range_reduction = True
    if plot_range == None:
        range_reduction = False
    elif not isinstance(plot_range, list):
        raise TypeError('The argument should be a list of floats.')
    elif len(plot_range) != 4:
        raise ValueError(
            'The number of elements in plot_range should be 4, here it is {}'.format(len(plot_range)))

    if q_x.shape != X.shape or q_x.shape != Y.shape or q_y.shape != X.shape or q_y.shape != Y.shape:
        raise ValueError("The shape of X, Y and the two elements in Z must coincide.")

    if range_reduction:
        x_max = plot_range[1]
        x_min = plot_range[0]
        y_max = plot_range[3]
        y_min = plot_range[2]

        idx_x_min, idx_x_max = _crop_array_idxs(X[:, 0], x_min, x_max)
        idx_y_min, idx_y_max = _crop_array_idxs(Y[0, :], y_min, y_max)

        X = X[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]
        Y = Y[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]
        q_x = q_x[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]
        q_y = q_y[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]

    # plotting of the function
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    Q = ax.quiver(X, Y, q_x, q_y, pivot='tail')
    ax.quiverkey(Q, 0.9, 0.9, 1, '1' + mes_unit, labelpos='E', coordinates='figure')

    if range_reduction:
        ax.axis(plot_range)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

    if show_plot:
        plt.show()

    return fig

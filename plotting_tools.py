import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.animation as animation


fig_size = (10, 6)
cm_surface = cm.get_cmap('coolwarm')
cm_pcolor = cm.get_cmap('Spectral')



def plane_plotter(x, y, title='', x_label=r'$x$', y_label=r'$y$', log_x=False, log_y=False, show_plot=True):
    """
    Generates a simple plot of the pairs of array x and y in a plane.

    Parameters
    ----------
    x : numpy.ndarray
        Values on the horizontal axis.
    y : numpy.ndarray
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    ax.plot(x, y)

    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if show_plot:
        plt.show()

    return fig



def pcolor_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$', show_plot=True):
    """
    Generates a plane figure where the third varaible is represented as shades
    of color.

    Parameters
    ----------
    x : numpy.ndarray
        Matrix with values for the first axis on all the rows.
    y : numpy.ndarray
        Matrix with values for the second axis on all the columns.
    z : numpy.ndarray
        Matrix with values of the third axis.
    title : str, optional
        Title of the plot. The default is ''.
    x_label : str, optional
        Name of the horizontal axis. The default is r'$x$'.
    y_label : TYPE, optional
        Name of the vertical axis. The default is r'$y$'.
    show_plot: bool, optional
        Flag for printing the figure with plt.show(). The default value is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    pc = ax.pcolor(x, y, z, cmap=cm_pcolor)
    fig.colorbar(pc, shrink=0.5, aspect=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if show_plot:
        plt.show()

    return fig



def surface_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$', z_label=r'$z$', show_plot=True):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with the plot.

    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.gca(projection='3d')

    s = ax.plot_surface(x, y, z, cmap=cm_surface, linewidth=0, antialiased=False)
    fig.colorbar(s, shrink=0.5, aspect=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label, rotation=90)
    ax.set_title(title)

    if show_plot:
        plt.show()

    return fig



def surface_animate(X, Y, Z, delay=200, title='', x_label=r'$x$', y_label=r'$y$', z_label=r'$z$', show_plot=True):
    """
    Animates a sequence of 3D data.

    Parameters
    ----------
    X : ndarray
        Matrix with values for the first axis on all the rows. The size should
        be the same as Y.
    Y : ndarry
        Matrix with values for the second axis on all the columns. The size should
        be the same as X.
    Z : ndarray
        Tensor containing the height of each point and each time. The first
        dimension is for the time step and Z[i,:] should have the same dimension
        as X and Y.
    delay : int
        Description of parameter `delay`. The default is 200.
    title : str
        Description of parameter `title`. The default is ''.
    x_label : str
        Description of parameter `x_label`. The default is r'$x$'.
    y_label : str
        Description of parameter `y_label`. The default is r'$y$'.
    z_label : str
        Description of parameter `z_label`. The default is r'$z$'.
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

    frames = []

    for t in range(Z.shape[0]):
        s = ax.plot_surface(X, Y, Z[t,:], cmap=cm_surface, linewidth=0, antialiased=False)
        frames.append([s])

    anim = animation.ArtistAnimation(fig, frames, interval=delay, \
                                     repeat_delay=2*delay, repeat=True, blit=False)

    if show_plot:
        plt.show()

    return anim

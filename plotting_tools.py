import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

fig_size = (10, 6)
cm_surface = cm.get_cmap('coolwarm')
cm_pcolor = cm.get_cmap('Spectral')

def plane_plotter(x, y, title='', x_label=r'$x$', y_label=r'$y$', log_x=False, log_y=False):
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

    plt.show()

    return fig


def pcolor_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$'):
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

    plt.show()

    return fig


def surface_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$', z_label=r'$z$'):
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

    plt.show()

    return fig



if __name__ == "__main__":
    import numpy as np

    x = np.linspace(-3, 3, 100)
    y = np.arctan(x)
    plane_plotter(x, y, title="Plane Plotter Test", x_label=r'$x$', y_label=r'$\arctan(x)$')

    X, Y = np.meshgrid(x, x)
    fun = lambda x, y : np.sin(np.sqrt(X**2 + Y**2))
    Z = fun(X, Y)

    pcolor_plotter(X, Y, Z, title="Pcolor Test")
    surface_plotter(X, Y, Z, title="Surface Plotter Test", z_label=r'$\sin(x^2 + y^2)$')

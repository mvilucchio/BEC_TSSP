import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#from mpl_toolkits import mplot3d
import matplotlib.animation as animation


fig_size = (10, 6)

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

    fig.patch.set_facecolor(blk_background)

    if isinstance(axes, np.ndarray):
        if len(axes.shape) == 1:
            axes = np.expand_dims(axes, 0)
    else:
        axes = np.array([[axes]])

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].set_facecolor(blk_background)

            axes[i,j].xaxis.label.set_color(text_color)
            axes[i,j].yaxis.label.set_color(text_color)

            axes[i,j].title.set_color(text_color)

            for _, spine in axes[i,j].spines.items():
                spine.set_color(text_color)

            for xline in axes[i,j].get_xgridlines():
                xline.set_color(grid_line_color)

            for yline in axes[i,j].get_ygridlines():
                yline.set_color(grid_line_color)

            for xticklab in axes[i,j].get_xticklabels():
                xticklab.set_color(text_color)

            for yticklab in axes[i,j].get_yticklabels():
                yticklab.set_color(text_color)

            for xtick in axes[i,j].get_xticklines():
                xtick.set_color(text_color)

            for ytick in axes[i,j].get_yticklines():
                ytick.set_color(text_color)



def plane_plotter(x_list, y_list, title='', x_label=r'$x$', y_label=r'$y$', log_x=False, log_y=False, show_plot=True, dark=False):
    """
    Generates a simple plot of the pairs of array x and y in a plane.

    Parameters
    ----------
    x_list : numpy.ndarray
        Values on the horizontal axis.
    y_list : numpy.ndarray or list
        Values on the vertical axis. Every element of the list should be a
        numpy.ndarray.
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

    Raises
    ------
    ValueError
        If the size of the given list don't match any of the case.

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
        raise ValueError('The number of elements for x_list and y_list should be the same or 1 for x_list.')

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



def contour_plotter(X, Y, Z, title='', x_label=r'$x$', y_label=r'$y$', show_plot=True, dark=False):


    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    cp = ax.contour(X, Y, Z, cmap=cm_contour)
    ax.clabel(cp, inline=1, fontsize=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = ax.set_title(title)

    if dark:
        _darkizer(fig, ax, title)

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



def quiver_plotter(x, y, z, title='', x_label=r'$x$', y_label=r'$y$', mes_unit='', show_plot=True):

    if isinstance(z, list):
        if len(z) != 2 or len(z[0].shape) != 2 or len(z[0].shape) != 2:
            raise ValueError("a")
        else:
            q_x = z[0]
            q_y = z[1]
    else:
        if len(z.shape) != 3:
            raise ValueError("a")
        else:
            q_x = z[0,:]
            q_y = z[1,:]

    if q_x.shape != x.shape or q_x.shape != y.shape or q_y.shape != x.shape or q_y.shape != y.shape:
        raise ValueError("a")

    fig = plt.figure(figsize=fig_size)
    ax = fig.gca()

    Q = ax.quiver(x, y, q_x, q_y, pivot='tail')
    ax.quiverkey(Q, 0.9, 0.9, 1, '1' + mes_unit, labelpos='E',
                   coordinates='figure')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if show_plot:
        plt.show()

    return fig

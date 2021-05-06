#!/usr/bin/env python
# coding: utf-8

# # Numerical Methods Project: Vorticies in BEC with Harmonic Trap

# The following inmplementation can be found on [GitHub](https://github.com/superporchetta/numerical_methods_project).

# In[ ]:



# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime


from matplotlib import cm
# custom libraries with all the fucntions used
import plotting_tools as pt
import gross_pitaevskii as gp


# All the function that are used to integrate the equations or to calculate phyisical meaningful quantities are contained in `gross_pitaevskii` while the file `plotting_tools` contains many useful functions to generate plots.
#
# For a detailed discussion of the function paramethers we advice to look at the documentation in the function definitions.
#
# In addition here we define some useful functions and variables for later use that will be used later on.

# In[ ]:


fx = lambda x, y: x
fy = lambda x, y: y
fx2 = lambda x, y: x**2
fy2 = lambda x, y: y**2


# Many results from this notebooks can be saved in a subdirectory of the cwd called `test_imgs`. The following lines of code create the directory.

# In[ ]:


img_path = "/data/imgs"
num_path = "/data/numerical"



# And the last thing needed is a function to save the numerical data

# In[ ]:


def saving_files(path, *arrays, msg=""):
    now = str(datetime.now())[:-7]
    now = now[:-15] + "_" + now[-14:-12] + "_" + now[-11:-9] + "_" + now[-8:-6] + "_" + now[-5:-3] + "_" + now[-2:]

    np.savez(path + "/" + msg + "_" + str(now), *arrays)


# ## Test of Algorithms from [1.]

# The following tests are run like the example **2.I** (pg. 332) in the above mentioned paper.
#
# In the use of this funciton the potential can depend on time (first and second argument are the coordinates and the third one is time). In the following exampe we are looking at the evolution of out initial condition on a constant potential (i.e. an harmonic trap).

# In[ ]:


x_range = [-8, 8]
eps = 1.0
beta = 2.0
dt = 0.001
dx = 1/32
T = 10.0
N = int(T / dt)
M = int((x_range[1]-x_range[0])/dx)
q = 40
Dt = q*dt

ho_potential = lambda x, y: 0.5*(x**2 + y**2)

def f0(x, y):
    res = np.zeros(x.shape, dtype=complex)
    res = np.exp(-(x**2 + y**2)/(2*eps))
    a = res/np.sqrt(np.pi*eps)
    return a

#f0 = lambda x, y: np.exp(-(x**2 + y**2)/(2*eps))/np.sqrt(np.pi*eps)

start_time = time.time()
t, X, Y, psi = gp.ti_tssp_2d_pbc(M, N, q, x_range, x_range, f0, ho_potential, dt, beta, eps)
end_time = time.time()
print("--- Evaluated in {:.2f} seconds ---".format(end_time - start_time))


# In[17]:

def _crop_array_idxs(array, val_min, val_max):
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


import matplotlib.animation as animation

x_max = 3.0
x_min = -3.0
y_max = 3.0
y_min = -3.0

idx_x_min, idx_x_max = _crop_array_idxs(X[:, 0], x_min, x_max)
idx_y_min, idx_y_max = _crop_array_idxs(Y[0, :], y_min, y_max)

X_c = X[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]
Y_c = Y[idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1]

fig, ax = plt.subplots()

mode = 'contour'
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are animating three artists, the contour and 2
# annotatons (title), in each frame
ims = []
for i in range(len(psi[:,0,0])):
    if mode == 'contour':
        im = ax.contourf(X_c, Y_c, np.abs(psi[i, idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1])**2)
        add_arts = im.collections
    elif mode == 'imshow':
        im = ax.imshow(np.abs(psi[i, idx_x_min: idx_x_max + 1, idx_y_min: idx_y_max + 1])**2, extent=[np.min(X_c), np.max(X_c),
                                               np.min(Y_c), np.max(Y_c)],
                       aspect='auto')
        add_arts = [im, ]

    text = 'title={0!r}'.format(i)
    te = ax.text(90, 90, text)
    an = ax.annotate(text, xy=(0.45, 1.05), xycoords='axes fraction')
    ims.append(add_arts + [te,an])

#ani = animation.ArtistAnimation(fig, ims, interval=70,repeat_delay=1000, blit=False)
ani = animation.ArtistAnimation(fig, ims)
## For saving the animation we need ffmpeg binary:
#FFwriter = animation.FFMpegWriter()
#ani.save('basic_animation.mp4', writer = FFwriter)
plt.show()

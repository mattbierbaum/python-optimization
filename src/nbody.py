import numpy as np
import scipy as sp

import matplotlib.pyplot as pl
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def initialize_random(N=1024, phi=0.5, radius=1.0):
    """
    Creates a random uniform (perhaps overlapping) distribution of particles
    inside a 2D box given a packing fraction

    Parameters:
    ----------- N : integer [default: 1024]
        the number of particles in the simulation (radius 1)

    phi : float 
        packing fraction to use while initializing the particles

    radius : float, ndarray
        radius of the particles in the simulation

    Returns:
    --------
    simulation : dictionary
    """
    box_side = (N*np.pi*radius**2 / phi)**(1./2)
    box = np.array([box_side, box_side])

    pos = box[None,:]*np.random.rand(N,2)
    rad = radius*np.ones(N)
    vel = 0*pos

    sim = {'N': N, 'pos': pos, 'rad': rad, 'vel': vel, 'box': box}
    return sim

def display_particles(sim, plot=None):
    """
    Display the current configuration of particles using mpl patches.
    If given a previous figure and patch list, update them and redraw
    to save some time.

    Parameters:
    -----------
    sim : dictionary (simulation)
        the simulation / particle configuration to display

    plot : list[`matplotlib.figure.Figure`, Circles]
        a figure to modify if we have a display environment

    Returns:
    --------
    plot : list[`matplotlib.figure.Figure`, Circles]
        a plotting environment that can be modified later
    """

    fresh = plot is None

    if fresh:
        fig = pl.figure(figsize=(10,10))
        fig.add_axes([0,0,1,1])
    else:
        fig, patches = plot

    x,y = sim['pos'].T
    rad = sim['rad']
    box = sim['box']

    ax = fig.axes[0]

    if fresh:
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(b=False, which='both', axis='both')

        patches = [Circle((i,j), s, color='#555555', alpha=0.5) for i,j,s in zip(x,y,rad)]
        [ax.add_patch(p) for p in patches]
    else:
        for patch,i,j in zip(patches, x, y):
            patch.center = (i,j)
    
    pl.draw()
    pl.show()

    return fig, patches

def force1(sim):
    N = sim['N']
    pos = sim['pos']

    f = np.zeros_like(pos)
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                rij = pos[i] - pos[j]
                dist = np.sqrt((rij**2).sum())

                if dist < 2.0:
                    f[i] = 100*(1-dist/2.0)**2 * rij/dist
    return f

def force2(sim):
    N = sim['N']
    pos = sim['pos']

    f = np.zeros_like(pos)
    for i in xrange(N):
        rij = pos[i] - pos
        dist = np.sqrt((rij**2).sum())

        if dist < 2.0:
            f[i] += 100*(1-rij/2.0)**2 * rij/dist

    return f

def step(sim, steps=1, dt=1e-2, plot=None, force_func=force1):
    for step in xrange(steps):
        sim['vel'] += force_func(sim)*dt
        sim['pos'] += sim['vel']*dt

        if plot is not None:
            display_particles(sim, plot)

        print step

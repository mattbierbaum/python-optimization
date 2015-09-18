import numpy as np
import scipy as sp

import matplotlib.pyplot as pl
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

PASSIVE = 0
ACTIVE = 1

class Moshpit(object):
    def __init__(self, N=500, phi=0.7, fraction=0.3, beta=1, epsilon=120, T=1, dt=1e-2):
        """
        Creates a moshpit simulation of active and passive moshers according to
        the paper arXiv:1302.1886.  By default, creates particles in a random
        uniform (perhaps overlapping) distribution of particles inside a 2D box
        given a packing fraction

        Parameters:
        -----------
        N : integer [default: 1024]
            the number of particles in the simulation (radius 1)

        phi : float 
            packing fraction to use while initializing the particles

        fraction : float
            fraction of the population that is active [0,1]

        beta : float
            damping parameter f = -\beta v

        epsilon : float
            force constant for the soft-sphere potential f = \epsilon (1-d/d_0)^{3/2}

        T : float
            temperature of the active participants

        dt : float
            timestep for the integrator
        """
        self.radius = 1.0
        self.N = N
        self.phi = phi
        self.fraction = fraction
        self.beta = beta
        self.epsilon = epsilon
        self.T = T
        self.dt = dt

        # find the box size based on the number of particles and packing fraction
        self.box_side = (self.N*np.pi*self.radius**2 / self.phi)**(1./2)
        self.box = np.array([self.box_side, self.box_side])

        self.init_random()

    def init_random(self):
        """
        Place the current particles into a stationary, random distribution with
        active and passive particles mixed randomly as well
        """
        self.pos = self.box[None,:]*np.random.rand(self.N,2)
        self.rad = self.radius*np.ones(self.N)
        self.vel = 0*self.pos
    
        # set the particle type randomly using uniform RNG
        self.typ = (np.random.rand(self.N) < self.fraction).astype('int')

    def init_circle(self):
        """
        Place the current particles into a stationary distribution in which
        active moshers are clustered in the centered in a circle in the simulation
        """
        self.init_random()

        # the radius of the circle given a given population fraction
        radius = np.sqrt(self.fraction * self.box_side**2 / np.pi)

        # need to find the particles in a circle of that size in the center
        # | pos - center | < radius
        self.typ = (np.sqrt(((self.pos-self.box/2)**2).sum(axis=-1)) < radius).astype('int')

    def force_damp(self):
        """ Calculate the damping force -beta v """
        return -self.beta * self.vel
    
    def force_noise(self):
        """ Calculate the effective force of the Langevin dynamics """
        coeff = np.sqrt(2*self.T*self.beta/self.dt)
        return coeff * np.random.randn(*self.pos.shape) * (self.typ == ACTIVE)[:,None]
    
    def boundary_condition(self):
        """ Apply hard reflective boundary conditions to particles """
        for i in xrange(2):
            mask = (self.pos[:,i] < 0)
            self.pos[mask,i] = 2*0-self.pos[mask,i]
            self.vel[mask,i] *= -1
    
            mask = (self.pos[:,i] > self.box[i])
            self.pos[mask,i] = 2*self.box[i]-self.pos[mask,i]
            self.vel[mask,i] *= -1
    
    def integrate(self, force_func=None):
        """
        Integrate the equations of motion. For this simple integrator, we are
        using the simplest sympletic integrator, NSV where

            v_{n+1} = v_n + f*dt
            x_{n+1} = x_n + v_{n+1}*dt

        Parameters:
        -----------
        force_func : function(self) -> forces (ndarray[N,2])
            a function which takes a Moshpit and returns the pair forces
            on each particle from the others.  separated from the class
            for pedagogy
        """
        self.forces = force_func(self) + self.force_damp() + self.force_noise()
        self.vel += self.forces*self.dt
        self.pos += self.vel*self.dt
    
    def step(self, steps=1, display_interval=20, plot=None, force_func=None):
        """ Perform a set of integration / BC steps and update plot """
        for step in xrange(steps):
            self.integrate(force_func)
            self.boundary_condition()
    
            if step % display_interval == 0 and plot is not None:
                plot.update(self)



class DisplayDiscs(object):
    def __init__(self, sim):
        """
        Display the current configuration of particles using mpl patches.  If
        given a previous figure and patch list, update them and redraw to save
        some time.

        Parameters:
        -----------
        sim : `Moshpit' object
            the simulation / particle configuration to display
        """
        pl.show(block=False)
        self.fig = pl.figure(figsize=(10,10))
        self.ax = self.fig.add_axes([0,0,1,1])

        x,y = sim.pos.T
        rad = sim.rad
        box = sim.box
        typ = sim.typ

        self.ax.set_xlim(0, box[0])
        self.ax.set_ylim(0, box[1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(b=False, which='both', axis='both')

        color_func = lambda t: '#555555' if t == PASSIVE else '#770000'
        colors = map(color_func, typ)

        self.patches = [Circle((i,j), s, color=c, alpha=0.5) for i,j,s,c in zip(x,y,rad,colors)]
        [self.ax.add_patch(p) for p in self.patches]
        
    def update(self, sim):
        """ Update the display with the simulation """
        x,y = sim.pos.T

        for patch,i,j in zip(self.patches, x, y):
            patch.center = (i,j)
        
        pl.draw()
        pl.show()

class DisplayPoints(object):
    def __init__(self, sim):
        """
        Display the current configuration of particles using plain 'ol line
        collections from pl.plot -- much faster than the patches

        Parameters:
        -----------
        sim : `Moshpit' object
            the simulation / particle configuration to display
        """
        pl.show(block=False)

        self.fig = pl.figure(figsize=(10,10))
        self.ax = self.fig.add_axes([0,0,1,1])

        x,y = sim.pos.T
        rad = sim.rad
        box = sim.box
        typ = sim.typ

        self.ax.set_xlim(0, box[0])
        self.ax.set_ylim(0, box[1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.grid(b=False, which='both', axis='both')

        act = typ == ACTIVE
        pas = typ == PASSIVE
        self.plots_passive, = self.ax.plot(x[pas],y[pas],'ko')
        self.plots_active, = self.ax.plot(x[act],y[act],'ro')

    def update(self, sim):
        """ Update the display with the simulation """
        x,y = sim.pos.T
        typ = sim.typ

        act = typ == ACTIVE
        pas = typ == PASSIVE
        self.plots_passive.set_data(x[pas],y[pas])
        self.plots_active.set_data(x[act],y[act])

        pl.draw()

def force1(sim):
    N = sim.N
    pos = sim.pos

    f = np.zeros_like(pos)
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                rij = pos[i] - pos[j]
                dist = np.sqrt((rij**2).sum())

                if dist < 2.0:
                    f[i] += sim.epsilon*(1-dist/2.0)**(1.5) * rij/dist

    return f

def force2(sim):
    N = sim.N
    pos = sim.pos

    f = np.zeros_like(pos)
    for i in xrange(N):
        rij = pos[i] - pos
        dist = np.sqrt((rij**2).sum(axis=-1))

        mask = (dist > 0)&(dist < 2)
        rij = rij[mask]
        dist = dist[mask][:,None]

        if len(rij) > 0:
            forces = sim.epsilon*(1-dist/2.0)**(1.5) * rij/dist
            f[i] += forces.sum(axis=0)

    return f

from math import sqrt
from numba import jit

@jit(nopython=True)
def _inner_naive(pos, N, eps, f):
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                x = pos[i,0] - pos[j,0]
                y = pos[i,1] - pos[j,1]
                dist = sqrt(x*x + y*y)

                if dist < 2.0:
                    c = eps*(1-dist/2.0)**(1.5)
                    f[i][0] += c * x/dist
                    f[i][1] += c * y/dist
    return f

def force3(sim):
    N = sim.N
    pos = sim.pos

    f = np.zeros_like(pos)
    return _inner_naive(pos, N, sim.epsilon, f)

@jit(nopython=True)
def _nbl_forces(cells, counts, nside, box, pos, N, eps, f):
    for i in xrange(N):
        ix = int((pos[i,0] / box[0]) * nside)
        iy = int((pos[i,1] / box[1]) * nside)

        c = counts[ix,iy]
        cells[ix,iy,c] = i
        counts[ix,iy] += 1

    for i in xrange(N):
        ix = int((pos[i,0] / box[0]) * nside)
        iy = int((pos[i,1] / box[1]) * nside)

        for tx in xrange(max(0,ix-1), min(ix+2, nside)):
            for ty in xrange(max(0,iy-1), min(iy+2, nside)):
                for c in xrange(counts[tx,ty]):
                    ind = cells[tx,ty,c]
                    x = pos[i,0] - pos[ind,0]
                    y = pos[i,1] - pos[ind,1]
                    dist = sqrt(x*x + y*y)

                    if dist < 2.0 and dist > 0:
                        c = eps*(1-dist/2.0)**(1.5)
                        f[i][0] += c * x/dist
                        f[i][1] += c * y/dist
    return f

def force4(sim):
    N = sim['N']
    pos = sim['pos']
    box = sim['box']

    f = np.zeros_like(pos)
    nside = int(box[0]/2.1)
    nside = nside if nside > 0 else 1

    cells = np.zeros((nside, nside, 10), dtype='int')
    counts = np.zeros((nside, nside), dtype='int')
    return _nbl_forces(cells, counts, nside, box, pos, N, sim['epsilon'], f)


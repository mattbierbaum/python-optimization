import display
import forces

import numpy as np
import scipy as sp

PASSIVE = 0
ACTIVE = 1

class Moshpit(object):
    def __init__(self, N=500, phi=0.7, fraction=0.18, beta=1, epsilon=120, T=1, dt=1e-2):
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
    
    def integrate(self, forces):
        """
        Integrate the equations of motion. For this simple integrator, we are
        using the simplest sympletic integrator, NSV where

            v_{n+1} = v_n + f*dt
            x_{n+1} = x_n + v_{n+1}*dt

        Parameters:
        -----------
        forces : ndarray[N,2]
            the forces on each particle
        """
        self.vel += forces*self.dt
        self.pos += self.vel*self.dt
    
    def step(self, steps=1, display_interval=20, disp=None, force_func=None):
        """
        Perform a set of integration / BC steps and update plot

        Parameters:
        -----------
        steps : int
            number of time steps of size self.dt to perform

        display_interval : int
            number time steps between display updates. this is introduced since
            the application is often draw limited

        disp : `display.Display' object
            the particular display to update (can be None)

        force_func : function
            a function which takes a Moshpit and returns the pair forces
            on each particle from the others.  separated from the class
            for pedagogy
        """
        for step in xrange(steps):
            self.forces = force_func(self) + self.force_damp() + self.force_noise()
            self.integrate(self.forces)
            self.boundary_condition()
    
            if step % display_interval == 0 and disp is not None:
                disp.update(self)

    def relax(self):
        """ Relax the current configuration using just pair wise forces (no noise) """
        for step in xrange(1000):
            self.forces = forces.force5(self) + self.force_damp()
            self.integrate(self.forces)
            self.boundary_condition()


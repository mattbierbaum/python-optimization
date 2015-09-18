import matplotlib.pyplot as pl
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import numpy as np
import scipy as sp

PASSIVE = 0
ACTIVE = 1

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
        #pl.show(block=False)
        self.fig = pl.figure(figsize=(6,6))
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

        color_func = lambda t: '#222222' if t == PASSIVE else '#DD0000'
        colors = map(color_func, typ)

        self.patches = [Circle((i,j), s, ec=None, fc=c, alpha=0.75) for i,j,s,c in zip(x,y,rad,colors)]
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
        #pl.show(block=False)

        # create a square, full-figure axis
        self.fig = pl.figure(figsize=(6,6))
        self.ax = self.fig.add_axes([0,0,1,1])

        # style the plot a bit cleaner
        self.ax.set_xlim(0, sim.box[0])
        self.ax.set_ylim(0, sim.box[1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(b=False, which='both', axis='both')

        # some nice local variables for typing
        x,y = sim.pos.T
        typ = sim.typ
        act = typ == ACTIVE
        pas = typ == PASSIVE

        # create two plots (line collections) for each type
        # so that we can have two colors
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



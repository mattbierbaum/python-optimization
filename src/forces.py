import numpy as np
import scipy as sp

from math import sqrt
from numba import jit

def force1(sim):
    N = sim.N
    pos = sim.pos
    dia = 2*sim.radius

    f = np.zeros_like(pos)
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                rij = pos[i] - pos[j]
                dist = np.sqrt((rij**2).sum())

                if dist < dia:
                    f[i] += sim.epsilon*(1-dist/dia)**2 * rij/dist

    return f

def force2(sim):
    N = sim.N
    pos = sim.pos
    dia = 2*sim.radius

    rij = pos[:,None,:] - pos[None,:,:]
    dist = np.sqrt((rij**2).sum(axis=-1))
    
    dist[np.eye(sim.N)==1.] = 1e3
    dist = dist[:,:,None]
    forces = (sim.epsilon*(1-dist/dia)**2 * rij/dist * (dist < dia)).sum(axis=1)

    return forces

def force3(sim):
    N = sim.N
    pos = sim.pos
    dia = 2*sim.radius

    f = np.zeros_like(pos)
    for i in xrange(N):
        rij = pos[i] - pos
        dist = np.sqrt((rij**2).sum(axis=-1))

        mask = (dist > 0)&(dist < dia)
        rij = rij[mask]
        dist = dist[mask][:,None]

        if len(rij) > 0:
            forces = sim.epsilon*(1-dist/dia)**2 * rij/dist
            f[i] += forces.sum(axis=0)

    return f

@jit(nopython=True)
def _inner_naive(pos, N, eps, f):
    for i in xrange(N):
        for j in xrange(N):
            if i != j:
                x = pos[i,0] - pos[j,0]
                y = pos[i,1] - pos[j,1]
                dist = sqrt(x*x + y*y)

                if dist < 2.0:
                    c = eps*(1-dist/2.0)**2
                    f[i][0] += c * x/dist
                    f[i][1] += c * y/dist
    return f

def force4(sim):
    force = np.zeros_like(sim.pos)
    return _inner_naive(sim.pos, sim.N, sim.epsilon, force)

@jit(nopython=True)
def _nbl_forces(cells, counts, nside, box, pos, N, eps, f):
    for i in xrange(N):
        ix = int((pos[i,0] / box[0]) * nside)
        iy = int((pos[i,1] / box[1]) * nside)

        p = counts[ix,iy]
        cells[ix,iy,p] = i
        counts[ix,iy] += 1

    for i in xrange(N):
        ix = int((pos[i,0] / box[0]) * nside)
        iy = int((pos[i,1] / box[1]) * nside)

        for tx in xrange(max(0,ix-1), min(ix+2, nside)):
            for ty in xrange(max(0,iy-1), min(iy+2, nside)):
                for p in xrange(counts[tx,ty]):
                    ind = cells[tx,ty,p]
                    x = pos[i,0] - pos[ind,0]
                    y = pos[i,1] - pos[ind,1]
                    dist = sqrt(x*x + y*y)

                    if dist < 2.0 and dist > 0:
                        c = eps*(1-dist/2.0)**2
                        f[i][0] += c * x/dist
                        f[i][1] += c * y/dist
    return f

def force5(sim):
    force = np.zeros_like(sim.pos)
    nside = int(sim.box[0]/(sim.radius*2.05))
    nside = nside if nside > 0 else 1

    cells = np.zeros((nside, nside, 10), dtype='int64')
    counts = np.zeros((nside, nside), dtype='int64')

    return _nbl_forces(cells, counts, nside, sim.box, sim.pos, sim.N, sim.epsilon, force)


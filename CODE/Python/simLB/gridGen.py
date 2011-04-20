# gridGen.py
"""
functions for generating adaptive rectilinear grids
"""

import numpy as np
import bezier as bz
from numpy import linalg

def a_progression(N, L, initial_spacing):
    """ arithematic progression, small at x = 0, with defined spacing"""
    if initial_spacing > L/float(N):
        initial_spacing = L/float(N)
        print 'error: initial spacing too large!!!'
        print 'set spacing to {}'.format(initial_spacing)
   

    dx_ = (2.0*(L-N*initial_spacing))/float((N*(N-1)))

    dx = np.zeros((N),dtype=float64)

    for i in range(N):
        dx[i] = initial_spacing + i*dx_
    
    return dx

def bezier_spacing(N, L, ptsx, ptsy, plot = 0):
    """setup spacing based on bezier curve, higher the value = tighter the spacing"""
    pts = []
    for i in range(len(ptsx)):
        pts.append(bz.pt(ptsx[i],ptsy[i]))
    
    bc = bz.Bezier(pts)
    dx = bz.xyBezier(bc,N)
    dx = 1.0/(dx + 1.0)
    dx = (L/sum(dx))*dx
    
    if plot > 0:
        bz.plotCP(pts)
        bz.plotBezier(bc,Nx)
        show()
    
    return np.array(dx)

def get_XY(dx, dy):
    """ given spacings, return the X & Y coords of the grid"""
    
    Nx = dx.size
    Ny = dy.size

    X = np.zeros((Nx+1),dtype=np.float64)
    Y = np.zeros((Ny+1),dtype=np.float64)
    
    for x in range(1,Nx+1):
        X[x] = X[x-1] + dx[x-1]
        
    for y in range(1,Ny+1):
        Y[y] = Y[y-1] + dy[y-1]
        
    return X, Y
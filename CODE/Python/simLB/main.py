# simLB
# 08/04/2011
"""
simLB is a Python port of my code for the 2D lattice Boltzmann method that
was originally coded in C++/CUDA
"""

#########################
## PACKAGES
#########################

import numpy as np

from enthought.mayavi import mlab

from LBM import ClassLBM

import time

import pyopencl as cl

#########################
## FUNCTIONS
#########################

def plotData(data_D, data_H):
    """
    plot passed in data as a surface
    """
    
    #plotting
    fig = mlab.figure(size = (512,512))
    mlab.view(90,0)
    
    cl.enqueue_read_buffer(lbm.queue, data_D, data_H).wait()
    
    X, Y = np.meshgrid(lbm.Y, lbm.X)
    Z = np.zeros((lbm.Nx, lbm.Ny))
    
    s = mlab.mesh(X,Y,Z, scalars = data_H, colormap = 'jet')
    
    #plot lines
    if 1:
        mlab.pipeline.surface(mlab.pipeline.extract_edges(s),
                              color=(0, 0, 0),line_width = 0.1, opacity = 0.1)

    return s

def plotUpdate(s, data_D, data_H):
    """
    update figure initialised by plotData()
    """
    
    cl.enqueue_read_buffer(lbm.queue, data_D, data_H).wait()
    
    s.mlab_source.scalars = data_H
    
#########################
## MAIN FUNCTIONALITY
#########################

# generate and initialise the primary simulation class
#  string indicates the initialisation script to use

#lbm = ClassLBM("makeQuarterCircle")
#lbm = ClassLBM("makeFullCircle")
#lbm = ClassLBM("makeLaxLiu3")
lbm = ClassLBM("makeBoundaryLayer")

#initialise plot
fig1 = plotData(lbm.rho_D, lbm.rho_H)
fig2 = plotData(lbm.T_D, lbm.T_H)
#fig3 = plotData(lbm.ux_D, lbm.ux_H)
#fig4 = plotData(lbm.uy_D, lbm.uy_H)

t0 = time.clock()

for i in range(lbm.steps):
    
    lbm.runSimStep()
    
    if i%lbm.nPrintOut == 0:
        plotUpdate(fig1, lbm.rho_D, lbm.rho_H)
        plotUpdate(fig2, lbm.T_D, lbm.T_H)
        #plotUpdate(fig3, lbm.ux_D, lbm.ux_H)
        #plotUpdate(fig4, lbm.uy_D, lbm.uy_H)

    print('step {}/{}'.format(i+1,lbm.steps))

t = time.clock() - t0

print('total time elapsed = {}'.format(t))

mlab.show()




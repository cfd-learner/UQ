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
from enthought.tvtk.api import tvtk

from LBM import ClassLBM

import time

import pyopencl as cl

#########################
## FUNCTIONS
#########################

def plotData(data_D, data_H, dataTitle):
    """
    plot passed in data as a surface
    """
    
    #plotting
    fig = mlab.figure(size = (512,512))
    mlab.view(90,0)
    
    cl.enqueue_read_buffer(lbm.queue, data_D, data_H).wait()
    
    rgrid = tvtk.RectilinearGrid()
    rgrid.cell_data.scalars = data_H.ravel()
    rgrid.cell_data.scalars.name = 'scalars'
    
    
    rgrid.dimensions = np.array((lbm.Ny+1, lbm.Nx+1, 1))
    rgrid.x_coordinates = lbm.Y
    rgrid.y_coordinates = lbm.X
    rgrid.z_coordinates = np.array([0.0])
    
    src = mlab.pipeline.add_dataset(rgrid)
    s = mlab.pipeline.cell_to_point_data(src)
    s = mlab.pipeline.surface(s)
    sb = mlab.scalarbar(s, title = dataTitle)
    
    #plot lines
    if 1:
        mlab.pipeline.surface(mlab.pipeline.extract_edges(src),
                              color=(0, 0, 0),line_width = 0.1, opacity = 0.05)

    return rgrid, src

def plotUpdate(rgrid, src, data_D, data_H):
    """
    update figure initialised by plotData()
    """
    
    cl.enqueue_read_buffer(lbm.queue, data_D, data_H).wait()
    
    rgrid.cell_data.scalars = data_H.ravel()
    rgrid.cell_data.scalars.name = 'scalars'
    rgrid.modified()
    src.update()
    
#########################
## MAIN FUNCTIONALITY
#########################

# generate and initialise the primary simulation class
#  string indicates the initialisation script to use

lbm = ClassLBM("makeQuarterCircle")
#lbm = ClassLBM("makeFullCircle")
#lbm = ClassLBM("makeLaxLiu3")
#lbm = ClassLBM("makeBoundaryLayer")

#initialise plot
grid1, src1 = plotData(lbm.rho_D, lbm.rho_H, 'density')
#grid2, src2 = plotData(lbm.T_D, lbm.T_H, 'temperature')
#grid3, src3 = plotData(lbm.ux_D, lbm.ux_H, 'velocity X')
#grid4, src4 = plotData(lbm.uy_D, lbm.uy_H, 'velocity Y')

t0 = time.clock()

for i in range(lbm.steps):
    
    lbm.runSimStep()
    
    if i%lbm.nPrintOut == 0:
        plotUpdate(grid1, src1, lbm.rho_D, lbm.rho_H)
        #plotUpdate(grid2, src2, lbm.T_D, lbm.T_H)
        #plotUpdate(grid3, src3, lbm.ux_D, lbm.ux_H)
        #plotUpdate(grid4, src4, lbm.uy_D, lbm.uy_H)

    print('step {}/{}'.format(i+1,lbm.steps))

t = time.clock() - t0

print('total time elapsed = {}'.format(t))

mlab.show()




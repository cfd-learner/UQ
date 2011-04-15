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

#########################
## MAIN FUNCTIONALITY
#########################

# generate and initialise the primary simulation class
#  string indicates the initialisation script to use

lbm = ClassLBM("makeQuarterCircle")

#plotting
fig = mlab.figure(size = (1000,1000))
data = mlab.pipeline.array2d_source(lbm.X,lbm.Y,lbm.rho_H)
s = mlab.pipeline.surface(data)
mlab.pipeline.surface(mlab.pipeline.extract_edges(data),
                            color=(0, 0, 0),line_width = 0.1)

t0 = time.clock()

for i in range(1000):
    lbm.runSimStep()
    
    s.mlab_source.scalars = lbm.rho_H

t = time.clock() - t0

print('total time elapsed = {}'.format(t))
    

    

# main loop
#while lbm.stop == 0:
#    lbm.runSimStep()

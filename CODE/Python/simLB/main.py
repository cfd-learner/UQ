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
x, y = np.mgrid[0:lbm.Nx:1,0:lbm.Ny:1]
fig = mlab.figure(size = (1000,1000))
s = mlab.surf(x, y, lbm.rho_H)


t0 = time.clock()

for i in range(10):
    lbm.runSimStep()
    
    s.mlab_source.scalars = lbm.rho_H

t = time.clock() - t0

print('total time elapsed = {}'.format(t))
    

    

# main loop
#while lbm.stop == 0:
#    lbm.runSimStep()

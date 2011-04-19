# makeQuarterCircle.py

"""
initialises all data for a simulation run 
boundary layer problem
"""

from numpy import *
from numpy import linalg
from pylab import *
import bezier as bz

class Setup:
    
    ##########################
    ## SIMULATION PARAMETERS
    
    R = 287.0       #gas constant J/kgK
    gamma = 1.4      # ratio of specific heats
    Pr = 0.72       #Prandtl number
    
    RKMETHOD = 1
    FMETHOD = 1
    CFL = 0.6
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0
    steps = 1
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    rho_ref = 0.0
    T_ref = 293.0
    S_v = 110.4
    nPrintOut = 1
    saveData = 0
    
    def initialise(self):
        ##########################
        ## domain definition
        
        self.Nx = 52     #number of elements in X
        self.Ny = 125    #number of elements in Y
        
        self.Lx = 1.0    #length of domain in X
        self.Ly = 0.68 #length of domain in y
        
        # grid generation                
        dx1 = 0.005 #self.Lx/float(self.Nx)
        dy1 = 0.005*self.Ly/float(self.Ny)
        
        if dx1 > self.Lx/float(self.Nx):
            dx1 = self.Lx/float(self.Nx)
            print 'error: x spacing too large!!!'
            print 'set x spacing to {}'.format(dx1)
        
        if dy1 > self.Ly/float(self.Ny):
            dy1 = self.Ly/float(self.Ny)
            print 'error: y spacing too large!!!'
            print 'set y spacing to {}'.format(dy1)
            
        
        dx_ = (2.0*(self.Lx-self.Nx*dx1))/(self.Nx*(self.Nx-1))
        dy_ = (2.0*(self.Ly-self.Ny*dy1))/(self.Ny*(self.Ny-1))
        
        self.dx = zeros((self.Nx),dtype=float64)
        self.dy = zeros((self.Ny),dtype=float64)
        
        for i in range(self.Nx):
            self.dx[i] = dx1 + i*dx_
            
        for i in range(self.Ny):
            self.dy[i] = dy1 + i*dy_
            
        #grid gen using bezier curve
        pts = [bz.pt(0,0), bz.pt(0.5,100), bz.pt(1.0,0)]
        bc = bz.Bezier(pts)
        dx = bz.xyBezier(bc,self.Nx)
        dx = 1.0/(dx + 1.0)
        dx = (self.Lx/sum(dx))*dx
        self.dx = dx
        
        bz.plotCP(pts)
        bz.plotBezier(bc,self.Nx)
        show()
        
        ##########################
        ## domain
        
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        #inlet & top boundary
        self.bnd[0,1:self.Ny] = 1
        self.bnd[0:self.Nx, self.Ny-1] = 1
        
        #wall
        self.bnd[0:self.Nx,0] = 2
        
        ##########################
        ## X and Y arrays - coordinates
        
        x_ = -self.dx[0]/2.0
        y_ = -self.dy[0]/2.0
        
        self.X = zeros((self.Nx),dtype=float64)
        self.Y = zeros((self.Ny),dtype=float64)
        
        for x in range(self.Nx):
            x_ += self.dx[x]
            self.X[x] = x_
            
        for y in range(self.Ny):
            y_ += self.dy[y]
            self.Y[y] = y_
                    
        
        ## VALUES FOR FLUID
        
        Re = 1.65e6
        
        rho0 = 0.0404
        T0 = 222.0
        p0 = rho0*self.R*T0
        ux0 = 2.0*sqrt(self.gamma*self.R*T0)
        uy0 = 0.0
        
        self.mu = (rho0*ux0*self.Lx)/Re
        
        
        
        ##########################
        ## SIMULATION LISTS
        self.numProps = 3
        self.density =   array([rho0, rho0, rho0],dtype=float64)
        self.therm =      array([T0, T0, T0],dtype=float64)
        self.velX = array([ux0, ux0, 0.0],dtype=float64)
        self.velY = array([uy0, uy0, 0.0],dtype=float64)
        self.cell =  array([0, 1, 2],dtype=uint32)
        
        if where(self.cell == 2):
            self.isSolid = 1
        else:
            self.isSolid = 0
        
        ##########################
        ## REFERENCE QUANTITIES
        
        self.rho_ref = rho0
        maxVel = sqrt(max(abs(self.velX))**2 + max(abs(self.velY))**2)
        self.Tc_Tref = (T0/self.T_ref)*(1.0 + (self.gamma - 1.0)/2.0)*(ux0/sqrt(self.gamma*self.R*T0))**2
        print('Tc_Tref = {}'.format(self.Tc_Tref))
        
# END
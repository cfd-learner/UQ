# makeQuarterCircle.py

"""
initialises all data for a simulation run 
boundary layer problem
"""

from numpy import *
from numpy import linalg

class Setup:
    
    ##########################
    ## SIMULATION PARAMETERS
    
    R = 287.0       #gas constant J/kgK
    gamma = 1.4      # ratio of specific heats
    mu = 1.86e-5    #dynamic viscosity
    Pr = 0.71       #Prandtl number
    
    RKMETHOD = 0
    FMETHOD = 0
    CFL = 0.1
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0.0
    steps = 10000
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    rho_ref = 0.0
    p_ref = 0.0
    T_ref = 0.0
    Tc_Tref = 3.0
    nPrintOut = 100
    
    def initialise(self):
        ##########################
        ## domain definition
        
        self.Nx = 50     #number of elements in X
        self.Ny = 100    #number of elements in Y
        
        self.Lx = 1.0    #length of domain in X
        self.Ly = 0.6 #length of domain in y
        
        #uniform spacing
        dx_ = self.Lx/self.Nx
        dy_ = self.Ly/self.Ny
        
        self.dx = dx_*ones((self.Nx),dtype=float64)
        self.dy = dy_*ones((self.Ny),dtype=float64)
        
        
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
            y_ = -self.dy[0]/2.0
            self.X[x] = x_
            for y in range(self.Ny):
                y_ += self.dy[y]
                self.Y[y] = y_
                    
        
        ## VALUES FOR FLUID
        
        rho0 = 1.165
        p0 = 101310.0
        T0 = p0/(rho0*self.R)
        ux0 = 2.0*sqrt(self.R*T0)
        uy0 = 0.0
        
        ##########################
        ## SIMULATION LISTS
        self.numProps = 3
        self.density =   array([rho0, rho0, rho0],dtype=float64)
        self.pressure =  array([p0, p0, p0],dtype=float64)
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
        self.p_ref = p0
        self.T_ref = self.p_ref/(self.rho_ref*self.R)
        
# END
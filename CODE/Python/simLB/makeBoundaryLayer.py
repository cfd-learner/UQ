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
    Pr = 0.72       #Prandtl number
    
    RKMETHOD = 1
    FMETHOD = 0
    CFL = 0.5
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0.01
    steps = 0
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    rho_ref = 0.0
    p_ref = 0.0
    T_ref = 293.0
    S_v = 110.4
    Tc_Tref = 2.0
    nPrintOut = 500
    saveData = 1
    
    def initialise(self):
        ##########################
        ## domain definition
        
        self.Nx = 52     #number of elements in X
        self.Ny = 125    #number of elements in Y
        
        self.Lx = 1.0    #length of domain in X
        self.Ly = 0.68 #length of domain in y
        
        # grid generation                
        dx1 = 0.0005 #self.Lx/float(self.Nx)
        dy1 = 0.005 #self.Ly/float(self.Ny)
        
        dx_ = (2.0*(self.Lx-self.Nx*dx1))/(self.Nx*(self.Nx-1))
        dy_ = (2.0*(self.Ly-self.Ny*dy1))/(self.Ny*(self.Ny-1))
        
        self.dx = zeros((self.Nx),dtype=float64)
        self.dy = zeros((self.Ny),dtype=float64)
        
        for i in range(self.Nx):
            self.dx[i] = dx1 + i*dx_
            
        for i in range(self.Ny):
            self.dy[i] = dy1 + i*dy_
        
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
        self.p_ref = self.rho_ref*self.R*self.T_ref;
        
# END
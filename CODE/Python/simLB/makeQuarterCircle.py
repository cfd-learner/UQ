# makeQuarterCircle.py

"""
initialises all data for a simulation run involving a quarter circle
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
    
    RKMETHOD = 1
    FMETHOD = 1
    CFL = 0.5
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0.0
    steps = 1000
    tol = 0.0
    dx = 0.0
    dy = 0.0
    Lx = 0.0
    Ly = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 1
    mirrorSouth = 1
    mirrorEast  = 1
    mirrorWest = 1
    rho_ref = 0.0
    p_ref = 0.0
    T_ref = 0.0
    Tc_Tref = 2.0
    nPrintOut = 100
    
    ##########################
    ## LISTS
    
    bnd = []
    
    density =   []
    pressure =  []
    therm =      []
    velX = []
    velY = []
    cell =  []
    

    
    def initialise(self):
        ##########################
        ## domain definition
        
        multi = 2.0
        
        self.Nx = 100     #number of elements in X
        self.Ny = int(ceil(multi*self.Nx))
        
        self.Lx = 0.5    #length of domain in X
        
        self.dx = self.Lx/(self.Nx-1)
        self.dy = self.dx
        
        self.Ly = (self.Ny - 1)*self.dy     #length of domain in y
        
        ##########################
        ## setup circle
        
        r = self.Lx/3.0
        
        B = zeros((self.Nx, self.Ny))
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        for x in range(self.Nx):
            x_ = x *self.dx
            for y in range(self.Ny):
                y_ = (y - self.Ny)*self.dy
                B[x,y] = linalg.norm([x_,y_])
                if B[x,y] < r:
                    self.bnd[x,y] = 1
                    
        
        #add solid block
        midX = floor(self.Nx/2)
        midY = floor(self.Ny/2)
        size = 3
        self.bnd[(midX-size):(midX+size),(midX-size):(midX+size)] = 2
        
        ## VALUES FOR CIRCLE
        
        rho0 = 1.165
        p0 = 101310.0
        T0 = p0/(rho0*self.R);
        
        rhoL = rho0
        TL = 303.0
        pL = rhoL*self.R*TL
        
        ## VALUES FOR BULK FLUID
        
        rhoR = 10.0*rho0
        TR = TL
        pR = rhoR*self.R*TR
        
        ##########################
        ## SIMULATION LISTS
        self.numProps = 2
        self.density =   array([rhoL,rhoR, rhoR],dtype=float64)
        self.pressure =  array([pL,pR, pR],dtype=float64)
        self.therm =      array([TL, TR, TR],dtype=float64)
        self.velX = array([0.0, 0.0, 0.0],dtype=float64)
        self.velY = array([0.0, 0.0, 0.0],dtype=float64)
        self.cell =  array([0, 0, 2],dtype=uint32)
        
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
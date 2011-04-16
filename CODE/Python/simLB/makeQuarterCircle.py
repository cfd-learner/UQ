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
        
        multi = 1.0
        
        self.Nx = 500     #number of elements in X
        self.Ny = 500     #number of elements in Y
        
        self.Lx = 0.5    #length of domain in X
        self.Ly = self.Lx*multi #length of domain in y
        
        #uniform spacing
        dx_ = self.Lx/self.Nx
        dy_ = self.Ly/self.Ny
        
        self.dx = dx_*ones((self.Nx),dtype=float64)
        self.dy = dy_*ones((self.Ny),dtype=float64)
        
        
        ##########################
        ## setup circle
        
        r = self.Lx/3.0
        
        B = zeros((self.Nx, self.Ny))
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
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
                B[x,y] = linalg.norm([x_,y_])
                self.Y[y] = y_
                if B[x,y] < r:
                    self.bnd[x,y] = 1
                    
        
        #add solid block
        #midX = floor(self.Nx/2)
        #midY = floor(self.Ny/2)
        #size = 3
        #self.bnd[(midX-size):(midX+size),(midX-size):(midX+size)] = 2
        
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
        self.numProps = 3
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
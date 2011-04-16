# makeQuarterCircle.py

"""
initialises all data for a simulation run 
Lax & Liu configuration 3
"""

from numpy import *

class Setup:
    
    ##########################
    ## SIMULATION PARAMETERS
    
    R = 287.0       #gas constant J/kgK
    gamma = 1.4      # ratio of specific heats
    mu = 1.86e-5    #dynamic viscosity
    Pr = 0.71       #Prandtl number
    
    RKMETHOD = 1
    FMETHOD = 1
    CFL = 0.2
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0.0
    steps = 1000
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    Tc_Tref = 2.5
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
        
        self.Nx = 200     #number of elements in X
        self.Ny = 200    #number of elements in Y
        
        self.Lx = 1.0     #length of domain in X
        self.Ly = self.Lx #length of domain in y
        
        #uniform spacing
        dx_ = self.Lx/self.Nx
        dy_ = self.Ly/self.Ny
        
        self.dx = dx_*ones((self.Nx),dtype=float64)
        self.dy = dy_*ones((self.Ny),dtype=float64)
        
        
        ##########################
        ## setup domain division
        
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        midX = self.Nx/2
        maxX = self.Nx
        midY = self.Ny/2
        maxY = self.Ny
        
        # domain 0, top right
        self.bnd[midX:maxX, midY:maxY] = 0
        
        # domain 1, top left
        self.bnd[0:midX, midY:maxY] = 1
        
        # domain 2, bottom left
        self.bnd[0:midX, 0:midY] = 2
        
        # domain 3, bottom right
        self.bnd[midX:maxX, 0:midY] = 3
        
        
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
                    
        
        ## VALUES FLUID
        
        #configuration
        cf = 3
        
        mrho = array([[1.5,1.1,1.0,1.0],[0.5323,0.5065,2.0,2.0],[0.138,1.1,1.0,1.0625],[0.5323,0.5065,3.0,0.5313]])
        mp = array([[1.5,1.1,1.0,1.0],[0.3,0.35,1.0,1.0],[0.029,1.1,1.0,0.4],[0.3,0.35,1.0,0.41]])
        mux = array([[0.0,0.0,0.75,0.0],[1.206,0.8939,0.75,0.0],[1.206,0.8939,-0.75,0.0],[0.0,0.0,-0.75,0.0]])
        muy = array([[0.0,0.0,-0.5,-0.3],[0.0,0.0,0.5,0.3],[1.206,0.8939,0.5,0.8145],[1.206,0.8939,-0.5,0.4276]])
        
        rho_ref = 1.165
        p_ref = 101310.0
        T_ref = p_ref/(rho_ref*self.R)
        u_ref = sqrt(self.R*T_ref)
        
        rho0 = mrho[0,cf]*rho_ref
        p0 =   mp[0,cf]*p_ref
        T0 =   p0/(rho0*self.R)
        ux0 =  mux[0,cf]*u_ref
        uy0 =  muy[0,cf]*u_ref
        
        rho1 = mrho[1,cf]*rho_ref
        p1 =   mp[1,cf]*p_ref
        T1 =   p1/(rho1*self.R)
        ux1 =  mux[1,cf]*u_ref
        uy1 =  muy[1,cf]*u_ref
        
        rho2 = mrho[2,cf]*rho_ref
        p2 =   mp[2,cf]*p_ref
        T2 =   p2/(rho2*self.R)
        ux2 =  mux[2,cf]*u_ref
        uy2 =  muy[2,cf]*u_ref
        
        rho3 = mrho[3,cf]*rho_ref
        p3 =   mp[3,cf]*p_ref
        T3 =   p3/(rho3*self.R)
        ux3 =  mux[3,cf]*u_ref
        uy3 =  muy[3,cf]*u_ref
        
        ##########################
        ## SIMULATION LISTS
        self.numProps = 4
        self.density =   array([rho0, rho1, rho2, rho3],dtype=float64)
        self.pressure =  array([p0, p1, p2, p3],dtype=float64)
        self.therm =      array([T0, T1, T2, T3],dtype=float64)
        self.velX = array([ux0, ux1, ux2, ux3],dtype=float64)
        self.velY = array([uy0, uy1, uy2, uy3],dtype=float64)
        self.cell =  array([0, 0, 0, 0],dtype=uint32)
        
        if where(self.cell == 2):
            self.isSolid = 1
        else:
            self.isSolid = 0
        
        ##########################
        ## REFERENCE QUANTITIES
        
        self.rho_ref = rho_ref
        self.p_ref = p_ref
        self.T_ref = T_ref
        
# END
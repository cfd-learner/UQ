# makeQuarterCircle.py

"""
initialises all data for a simulation run involving a quarter circle
"""

from numpy import *
from numpy import linalg
import gridGen as gg

class Setup:
    
    ##########################
    ## SIMULATION PARAMETERS
    
    R = 287.0       #gas constant J/kgK
    gamma = 1.4      # ratio of specific heats
    Pr = 0.71       #Prandtl number
    
    RKMETHOD = 1
    FMETHOD = 1
    CFL = 0.3
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0.0
    steps = 10000
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 1
    mirrorSouth = 1
    mirrorEast  = 1
    mirrorWest = 1
    
    nPrintOut = 50
    saveData = 0
    
    mu_model = 'VHS'    #GHS, sutherland, const, VHS
    if mu_model == 'sutherland':
        mu_ref = 1.71e-5
        T_ref = 273.0
        S_v = 110.4
    elif mu_model == 'GHS':
        ups1 = 2.0/13.0
        ups2 = 14.0/13.0
        phi = 0.61
        #gas props
        m = 66.3e-27    #kg
        sigma0 = 6.457e-19  #m^2
        mu_ref = 2.272e-5  #N/ms
        T_ref = 300 #K
    elif mu_model == 'VHS':
        T_ref = 273.0   #K
        mu_ref = 1.71e-5  #Ns/m**2 Appendix A: Bird
        upsilon = 1.0/6.0
    
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
        
        ##########################
        ## domain definition
        
        multi = 1.0
        
        self.Nx = 50     #number of elements in X
        self.Ny = int(self.Nx*multi)    #number of elements in Y
        
        self.Lx = 0.5    #length of domain in X
        self.Ly = multi*self.Lx #length of domain in y
        
        ptsx = [0.0, 0.25, 0.5, 0.75, 1.0]
        ptsy = [5.0, 5.0, 2.0, 1.0, 0.0]
        self.dx = gg.bezier_spacing(self.Nx, self.Lx, ptsx, ptsy)
        self.dy = gg.bezier_spacing(self.Ny, self.Ly, ptsx, ptsy)
        
        ##########################
        ## X and Y arrays - coordinates
        self.X, self.Y = gg.get_XY(self.dx, self.dy)
        
        ##########################
        ## setup circle
        
        r = self.Lx/3.0
        
        B = zeros((self.Nx, self.Ny))
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        x_ = -self.dx[0]/2.0
        
        for x in range(self.Nx):
            x_ += self.dx[x]
            y_ = -self.dy[0]/2.0
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
        u0 = sqrt(self.gamma*self.R*T0)
        
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
        
        self.ref_rho = rho0
        self.ref_T = T0
        if self.mu_model == 'sutherland':
            self.ref_mu = self.mu_ref*(T0/self.T_ref)**(3.0/2.0)*((self.T_ref+self.S_v)/(T0 + self.S_v))
        if self.mu_model == 'VHS':
            self.ref_mu = self.mu_ref*(T0/self.T_ref)**(0.5 + self.upsilon)
        
        self.Tc = T0*(1.0 + (self.gamma - 1.0)/2.0)*(u0/sqrt(self.gamma*self.R*T0))**2   #stagnation temp
        print('Tc = {}K'.format(self.Tc))
        
# END
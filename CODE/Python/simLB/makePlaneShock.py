# makeQuarterCircle.py

"""
initialises all data for a simulation run 
boundary layer problem
"""

from numpy import *
from numpy import linalg
from pylab import *
import bezier as bz
import gridGen as gg

class Setup:
    
    ##########################
    ## SIMULATION PARAMETERS
    
    R = 287.0       #gas constant J/kgK
    gamma = 5.0/3.0      # ratio of specific heats
    Pr = 0.72       #Prandtl number
    
    RKMETHOD = 1
    FMETHOD = 1
    CFL = 0.6
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0
    steps = 1000
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    
    nPrintOut = 100
    saveData = 0
    
    mu_model = 'GHS'    #GHS, sutherland, const
    if mu_model == 'sutherland':
        T_ref = 293.0
        S_v = 110.4
    elif mu_model == 'GHS':
        ups1 = 2.0/13.0
        ups2 = 14.0/13.0
        phi = 0.61
        #gas props
        m = 66.3e-27    #kg
        sigma0 = 6.457e-19  #m^2
        mu0 = 2.272e-5  #N/ms
        T_ref = 300 #K
    
    def initialise(self):
        ##########################
        ## domain definition
        
        # GAS PROPERTIES 1
        T1 = 150.0
        mu1 = 1.237e-5
        M1 = 4.0
        
        # GAS PROPERTIES 2
        T2 = T1*((2.0+(gamma-1.0)*M1**2)*((2*gamma*M1**2-(gamma-1.0))/((gamma+1.0)**2*M1**2)))
        
        g0 = (4.0*self.R*self.T_ref)**0.5
        S = S0*(T_ref/T2)**(ups2-ups1)
        
        mu2 = 
        
        # mean free path
        lambda1 = (2.0*mu)/(rho0*sqrt((8*R*T0)/pi))
        
        self.Nx = 100     #number of elements in X
        self.Ny = 2    #number of elements in Y
        
        self.Lx = 1.0    #length of domain in X
        self.Ly = 0.68 #length of domain in y
        
        ptsx = [0.0, 0.25, 0.5, 0.75, 1.0]
        ptsy = [10.0, 5.0, 2.0, 1.0, 0.0]
        self.dx = gg.bezier_spacing(self.Nx, self.Lx, ptsx, ptsy)
        self.dy = gg.bezier_spacing(self.Ny, self.Ly, ptsx, ptsy)
        
        ##########################
        ## X and Y arrays - coordinates
        self.X, self.Y = gg.get_XY(self.dx, self.dy)
        
        ##########################
        ## domain
        
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        #inlet & top boundary
        self.bnd[0,1:self.Ny] = 1
        self.bnd[0:self.Nx, self.Ny-1] = 1
        
        #wall
        self.bnd[0:self.Nx,0] = 2
        
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
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
    CFL = 0.3
    dtau = 0.0
    tt_tref = 0.0
    tt_time = 0
    steps = 100
    tol = 0.0
    periodicX = 0
    periodicY = 0
    mirrorNorth = 0
    mirrorSouth = 0
    mirrorEast  = 0
    mirrorWest = 0
    
    nPrintOut = 10
    saveData = 0
    
    mu_model = 'VHS'    #GHS, sutherland, const, VHS
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
        mu_ref = 2.272e-5  #N/ms
        T_ref = 300 #K
    elif mu_model == 'VHS':
        T_ref = 273.0   #K
        mu_ref = 2.117e-5  #Ns/m**2 Appendix A: Bird
        upsilon = 1.0/6.0
    
    def initialise(self):
        ##########################
        ## domain definition
        
        # GAS PROPERTIES 1
        rho1 = 1.0
        T1 = 150.0
        p1 = rho1*self.R*T1
        mu1 = self.mu_ref*(T1/self.T_ref)**(0.5 + self.upsilon)
        M1 = 4.0
        a1 = sqrt(self.gamma*self.R*T1)
        ux1 = M1*a1
        uy1 = 0.0
        
        # GAS PROPERTIES 2
        rho2 = rho1*((self.gamma + 1.0)*M1**2)/((self.gamma - 1.0)*M1**2+2.0)
        T2 = T1*((2.0+(self.gamma-1.0)*M1**2)*((2*self.gamma*M1**2-(self.gamma-1.0))/((self.gamma+1.0)**2*M1**2)))
        p2 = rho2*self.R*T2
        mu2 = self.mu_ref*(T2/self.T_ref)**(0.5 + self.upsilon)
        M2 = sqrt(((self.gamma-1.0)*M1**2+2.0)/(2*self.gamma*M1**2-(self.gamma-1.0)))
        a2 = sqrt(self.gamma*self.R*T2)
        ux2 = M2*a2
        uy2 = 0.0
        
        # mean free path
        lambda1 = (2.0*mu1)/(rho1*sqrt((8.0*self.R*T1)/pi))
        
        self.Nx = 100     #number of elements in X
        self.Ny = 10    #number of elements in Y
        
        m = 30.0    #multiplier of mean free path
        
        self.Lx = m*lambda1    #length of domain in X
        self.Ly = self.Lx #length of domain in y
        
        ptsxX = [0.0, 1.0]
        ptsyX = [0.0, 0.0]
        self.dx = gg.bezier_spacing(self.Nx, self.Lx, ptsxX, ptsyX)
        ptsxY = [0.0, 1.0]
        ptsyY = [0.0, 0.0]
        self.dy = gg.bezier_spacing(self.Ny, self.Ly, ptsxY, ptsyY)
        
        ##########################
        ## X and Y arrays - coordinates
        self.X, self.Y = gg.get_XY(self.dx, self.dy)
        
        ##########################
        ## domain
        
        self.bnd = zeros((self.Nx, self.Ny),dtype=uint32)
        
        midX = int(self.Nx/2.0)
        
        #inlet
        self.bnd[0,:] = 0
        
        #zone 1
        self.bnd[1:midX,:] = 1
        
        #zone 2
        self.bnd[midX:-1,:] = 2
        
        #outlet
        self.bnd[-1,:] = 3
        
        
        ##########################
        ## SIMULATION LISTS
        self.numProps = 4
        self.density =   array([rho1, rho1, rho2, rho2],dtype=float64)
        self.therm =      array([T1, T1, T2, T2],dtype=float64)
        self.velX = array([ux1, ux1, ux2, ux2],dtype=float64)
        self.velY = array([uy1, uy1, uy2, uy2],dtype=float64)
        self.cell =  array([1,0,0,1],dtype=uint32)
        
        if where(self.cell == 2):
            self.isSolid = 1
        else:
            self.isSolid = 0
        
        ##########################
        ## REFERENCE QUANTITIES
        
        self.ref_rho = rho1
        self.ref_T = T1
        self.ref_mu = mu1
        
        maxVel = sqrt(max(abs(self.velX))**2 + max(abs(self.velY))**2)
        self.Tc = 1.5*T1*(1.0 + (self.gamma - 1.0)/2.0)*(ux1/sqrt(self.gamma*self.R*T1))**2   #stagnation temp
        print('Tc = {}K'.format(self.Tc))
        
# END
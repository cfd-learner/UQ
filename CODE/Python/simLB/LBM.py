"""
LBM

class containing all methods required for simulation of the lattice Boltzmann
method
"""

from math import *
import numpy as np

import pylab as plt

# OpenCL
import pyopencl as cl

class ClassLBM:
    """ A class for containing all data and methods for LBM simulation"""
    
    ##########################
    ## SIMULATION PARAMETERS
    
    Ni = 13
    
    ##########################
    ## LISTS
    
    ex = np.array([0.0,1.0,0.0,-1.0,0.0,1.0,-1.0,-1.0,1.0,2.0,0.0,-2.0,0.0])    # x component of distribution function velocities
    ey = np.array([0.0,0.0,1.0,0.0,-1.0,1.0,1.0,-1.0,-1.0,0.0,2.0,0.0,-2.0])    # y component of dist. func. velocities
    
    ##########################
    ## SIMULATION VARIABLES
    
    dt = 0.0
    
    pause = 0
    stop = 0
    
    nsteps = 0
    step = 0
    ttotal = 0.0    #total time of simulation
    
    residual = 0.0  #residual of solution
    
    hasFailed = 0   #failure flag
    
        
    ##########################
    ## FUNCTIONS
    ##########################    
    def __init__(self, input):
        
        """
        perform initialisation of the class
        """
        
        ##########################
        ## OpenCL
        
        # select device
        for found_platform in cl.get_platforms():
            if found_platform.name == 'ATI Stream':
                my_platform = found_platform;
        
        for found_device in my_platform.get_devices():
            if found_device.name[0:5] == 'Intel':
                device = found_device
        
        self.ctx = cl.Context([device])
        #self.ctx = cl.create_some_context();
        self.queue = cl.CommandQueue(self.ctx)
        
        ##########################
        ## IMPORT
        #import the python script that defines the setup and initial conditions
        #of the run
        
        input = __import__(input)
        
        # run initialise script to define all variables
        simSetup = input.Setup()
        simSetup.initialise()
        
        ## Now import all variables into this class
        self.Nx = simSetup.Nx
        self.Ny = simSetup.Ny
        self.Lx = simSetup.Lx
        self.Ly = simSetup.Ly
        self.dx = simSetup.dx
        self.dy = simSetup.dy
        self.X = simSetup.X
        self.Y = simSetup.Y
        
        self.R = simSetup.R       #gas constant J/kgK
        self.gamma = simSetup.gamma      # ratio of specific heats
        self.mu = simSetup.mu    #dynamic viscosity
        self.Pr = simSetup.Pr       #Prandtl number
        
        self.RKMETHOD = simSetup.RKMETHOD
        self.FMETHOD = simSetup.FMETHOD
        self.CFL = simSetup.CFL
        self.dtau = simSetup.dtau
        self.tt_tref = simSetup.tt_tref
        self.tt_time = simSetup.tt_time
        self.steps = simSetup.steps
        self.tol = simSetup.tol
        self.periodicX = simSetup.periodicX
        self.periodicY = simSetup.periodicY
        self.mirrorNorth = simSetup.mirrorNorth
        self.mirrorSouth = simSetup.mirrorSouth
        self.mirrorEast  = simSetup.mirrorEast
        self.mirrorWest = simSetup.mirrorWest
        self.rho_ref = simSetup.rho_ref
        self.p_ref = simSetup.p_ref
        self.T_ref = simSetup.T_ref
        self.Tc_Tref = simSetup.Tc_Tref
        self.nPrintOut = simSetup.nPrintOut
        self.isSolid = simSetup.isSolid
        
        ##########################
        ## LISTS
        
        self.bnd = simSetup.bnd
        self.cell =  simSetup.cell
        
        self.numProps = simSetup.numProps
        self.density =   simSetup.density
        self.pressure =  simSetup.pressure
        self.therm =      simSetup.therm
        self.velX = simSetup.velX
        self.velY = simSetup.velY
        
        
        
        ##########################
        ## PLOTTING
        

        #plt.contour(self.bnd)
        #plt.show()
        
        
        ##########################
        ## REFINE INPUT
        
        self.refineInput()
        self.setupVars()
        
        ##########################
        ## INITIALISE DIST. FUNCTIONS
        self.initFunctions()
        
        #plt.contour(self.rho_H)
        #plt.show()
        
        # and you're good to go!
    
    def refineInput(self):
        """
        refine the input variables so that all required fields are populated
        with correct data
        """
        
        # define reference values
        if self.rho_ref == 0.0:
            self.rho_ref = self.p_ref/(self.R*self.T_ref)
        elif self.p_ref == 0.0:
            self.p_ref = self.rho_ref*self.R*self.T_ref
        elif self.T_ref == 0.0:
            self.T_ref = self.p_ref/(self.rho_ref*self.R)
        
        # update flow properties
        for r in range(self.numProps):
            if self.density[r] == 0.0:
                self.density[r] = self.pressure[r]/(self.R * self.therm[r])
            elif self.pressure[r] == 0.0:
                self.pressure[r] = self.density[r] * self.R * self.therm[r]
            elif self.therm[r] == 0.0:
                self.therm[r] = self.pressure[r]/(self.density[r] * self.R)
   
    def setupVars(self):
        """
        setup all of the variables as required
        """
        
        # compressible gas variables
        self.D = 2.0;    #dimensionality of sim
        self.b = 2.0/(self.gamma -1.0)   #total number of D.o.F
        self.K = self.b - self.D       # number of D.o.F to solve for
        
        # reference values
        self.Tc = self.Tc_Tref*self.T_ref      # characteristic temperature, K
        self.tau_ref = self.mu/self.p_ref;      # reference relaxation time
        self.u_ref = sqrt(self.R * self.T_ref)  #reference velocity
        self.t_ref = max(self.Lx,self.Ly)/self.u_ref    # reference time
        
        # simulation variables
        self.c = sqrt(self.R * self.Tc)     # characteristic sound speed, m/s
        
        # expand velocity lattice using chracteristic temperature
        self.ex = self.ex*self.c;
        self.ey = self.ey*self.c;
        
        # stability requirements
        
        if self.dt == 0:
            self.dt = self.CFL * min(np.min(self.dx), np.min(self.dy)) / np.max(self.ex)
        else:
            self.dt = self.dtau * self.tau_ref
        
        # simulation parameters
        need_steps = 0
        if self.tt_tref > 0.0:
            self.ttotal = self.tt_tref * self.t_ref
            need_steps = 1
        elif self.tt_time > 0:
            self.ttotal = self.tt_time
            need_steps = 1
        elif self.steps > 0:
            self.nsteps = self.steps
        elif self.tol > 0:
            self.nsteps = 0
        
        if need_steps == 1:
            self.nsteps = int(np.ceil(self.ttotal / self.dt))
    
    def initFunctions(self):
        """
        initialise all distribution functions
        setup all data storage on the device - depends upon method used (RK1/RK3)
        """
        
        mf = cl.mem_flags
        
        ##################
        ## host side data
        
        # global
        self.f_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #density dist. func.
        self.h_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #energy dist. func.
        
        self.rho_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   # macro. density
        self.ux_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   #macro. velocity x
        self.uy_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   #macro. velocity y
        self.T_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)     #macro temp.
        
        # local
        self.fr1_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #density dist. func.
        self.hr1_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #energy dist. func.
        
        if self.RKMETHOD == 0:
            self.fr1_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for f
            self.fr1_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for f
            self.hr1_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for h
            self.hr1_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for h
        
        elif self.RKMETHOD == 1:
            self.fr2_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #density dist. func.
            self.hr2_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #energy dist. func.
            
            self.fr3_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #density dist. func.
            self.hr3_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64)  #energy dist. func.
            
            self.rho3_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   # macro. density
            self.ux3_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   #macro. velocity x
            self.uy3_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)   #macro. velocity y
            self.T3_H = np.zeros((self.Nx,self.Ny),dtype=np.float64)     #macro temp.
            
            self.fr2_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for f
            self.fr2_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for f
            self.hr2_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for h
            self.hr2_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for h
            
            self.fr3_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for f
            self.fr3_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for f
            self.hr3_flux_x_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # x direction fluxes for h
            self.hr3_flux_y_H = np.zeros((self.Nx,self.Ny,self.Ni),dtype=np.float64) # y direction fluxes for h

        
        ######################
        ## OpenCL buffers
        
        # global
        self.f_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.f_H)
        self.h_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.h_H)
        
        self.dx_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.dx)
        self.dy_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.dy)
        
        self.rho_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.rho_H)
        self.ux_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.ux_H)
        self.uy_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.uy_H)
        self.T_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.T_H)
        
        self.bnd_D = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR,hostbuf = self.bnd)
        self.cell_D = cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR,hostbuf = self.cell)
        
        self.density_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.density)
        self.velX_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.velX)
        self.velY_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.velY)
        self.therm_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.therm)
        
        # local
        self.fr1_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr1_H)
        self.hr1_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr1_H)
        
        if self.RKMETHOD == 0:
            self.fr1_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr1_flux_x_H)
            self.fr1_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr1_flux_y_H)
            self.hr1_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr1_flux_x_H)
            self.hr1_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr1_flux_y_H)
        
        elif self.RKMETHOD == 1:
            self.fr2_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr2_H)
            self.hr2_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr2_H)
            
            self.fr3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr3_H)
            self.hr3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr3_H)
        
            self.rho3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.rho3_H)
            self.ux3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.ux3_H)
            self.uy3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.uy3_H)
            self.T3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.T3_H)
            
            self.fr2_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr2_flux_x_H)
            self.fr2_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr2_flux_y_H)
            self.hr2_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr2_flux_x_H)
            self.hr2_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr2_flux_y_H)
            
            self.fr3_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr3_flux_x_H)
            self.fr3_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.fr3_flux_y_H)
            self.hr3_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr3_flux_x_H)
            self.hr3_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.USE_HOST_PTR,hostbuf = self.hr3_flux_y_H)
            
        ##########################
        ## PACK DICTIONARY
        
        input = {'Nx':self.Nx,'Ny':self.Ny,'Ni':self.Ni,'R':self.R,'K':self.K,\
                 'b':self.b,'Tc':self.Tc,'dt':self.dt,\
                 'periodicX':self.periodicX,\
                 'periodicY':self.periodicY,'mirrorN':self.mirrorNorth,\
                 'mirrorS':self.mirrorSouth,'mirrorE':self.mirrorEast,\
                 'mirrorW':self.mirrorWest,'FMETHOD':self.FMETHOD,\
                 'mu':self.mu,'Pr':self.Pr,'isSolid':self.isSolid,\
                 'RKMETHOD':self.RKMETHOD}
        
        ##########################
        ## GENERATE ENTIRE OPENCL CODE HERE
        import codeCL
        name = codeCL.genOpenCL(input,self.ex,self.ey, self.dx, self.dy)
        f = open(name,'r')
        fstr = "".join(f.readlines())
        f.close()
        
        self.prg = cl.Program(self.ctx, fstr).build()
        print("OpenCL code compiled")
        
        #run the program
        self.global_size = self.rho_H.shape
        self.local_size = None
        
        # initialise all distriution functions
        self.prg.initFunctions(self.queue, self.global_size, self.local_size,
                               self.f_D, self.h_D, self.bnd_D, self.density_D,
                               self.velX_D, self.velY_D, self.therm_D,
                               self.rho_D, self.ux_D, self.uy_D, self.T_D)
        
        self.queue.finish()
        print("distribution functions initialised")
    
    def RK1(self):
        """
        perform one iteration of the Runge-Kutta one step method
        """
        
        # first step
        self.prg.RK1_STEP1(self.queue, self.global_size, self.local_size,
                           self.f_D, self.fr1_D, self.h_D, self.hr1_D,
                           self. cell_D, self.bnd_D, self.therm_D,
                           self.velX_D, self.velY_D, self.rho_D, self.ux_D,
                           self.uy_D, self.T_D).wait()
        
        # perform flux routine
        self.prg.GLOBAL_FLUXES(self.queue, self.global_size, self.local_size,
                               self.fr1_D, self.hr1_D, self.cell_D, self.bnd_D,
                               self.fr1_flux_x_D, self.fr1_flux_y_D,
                               self.hr1_flux_x_D, self.hr1_flux_y_D).wait()
        
        if self.isSolid == True:
            self.prg.WALL_FLUXES(self.queue, self.global_size, self.local_size,
                                 self.fr1_flux_x_D, self.fr1_flux_y_D,
                                 self.hr1_flux_x_D, self.hr1_flux_y_D,
                                 self.cell_D, self.bnd_D, self.therm_D,
                                 self.velX_D, self.velY_D).wait()
        
        # perform combination step
        self.prg.RK1_COMBINE(self.queue, self.global_size, self.local_size,
                             self.f_D, self.fr1_D, self.h_D, self.hr1_D,
                             self. cell_D, self.bnd_D, self.therm_D,
                             self.velX_D, self.velY_D, self.rho_D, self.ux_D,
                             self.uy_D, self.T_D, self.fr1_flux_x_D,
                             self.fr1_flux_y_D, self.hr1_flux_x_D,
                             self.hr1_flux_y_D).wait()
        
    def RK3(self):
        """
        perform one iteration of the Runge-Kutta one step method
        """
        
        # first step
        self.prg.RK3_STEP1(self.queue, self.global_size, self.local_size,
                           self.f_D, self.fr1_D, self.h_D, self.hr1_D,
                           self. cell_D, self.bnd_D, self.therm_D,
                           self.velX_D, self.velY_D, self.rho_D, self.ux_D,
                           self.uy_D, self.T_D).wait()
        
        # second step
        self.prg.RK3_STEP2(self.queue, self.global_size, self.local_size,
                           self.f_D, self.fr1_D, self.fr2_D, self.h_D,
                           self.hr1_D, self.hr2_D, self.cell_D, self.bnd_D,
                           self.therm_D, self.velX_D, self.velY_D, self.rho_D,
                           self.ux_D, self.uy_D, self.T_D).wait()
        
        # perform flux routine number 1
        self.prg.GLOBAL_FLUXES(self.queue, self.global_size, self.local_size,
                               self.fr2_D, self.hr2_D, self.cell_D, self.bnd_D,
                               self.fr2_flux_x_D, self.fr2_flux_y_D,
                               self.hr2_flux_x_D, self.hr2_flux_y_D).wait()
        
        if self.isSolid == True:
            self.prg.WALL_FLUXES(self.queue, self.global_size, self.local_size,
                                 self.fr2_flux_x_D, self.fr2_flux_y_D,
                                 self.hr2_flux_x_D, self.hr2_flux_y_D,
                                 self.cell_D, self.bnd_D, self.therm_D,
                                 self.velX_D, self.velY_D).wait()
        
        # third step
        self.prg.RK3_STEP3(self.queue, self.global_size, self.local_size,
                           self.f_D, self.fr2_D, self.fr3_D,
                           self.h_D, self.hr2_D, self.hr3_D,
                           self.cell_D, self.bnd_D, self.therm_D, self.velX_D,
                           self.velY_D, self.rho_D, self.rho3_D, self.ux_D,
                           self.ux3_D, self.uy_D, self.uy3_D, self.T_D,
                           self.T3_D, self.fr2_flux_x_D, self.fr2_flux_y_D,
                           self.hr2_flux_x_D, self.hr2_flux_y_D).wait()
        
        # perform flux routine number 2
        self.prg.GLOBAL_FLUXES(self.queue, self.global_size, self.local_size,
                               self.fr3_D, self.hr3_D, self.cell_D, self.bnd_D,
                               self.fr3_flux_x_D, self.fr3_flux_y_D,
                               self.hr3_flux_x_D, self.hr3_flux_y_D).wait()
        
        if self.isSolid == True:
            self.prg.WALL_FLUXES(self.queue, self.global_size, self.local_size,
                                 self.fr3_flux_x_D, self.fr3_flux_y_D,
                                 self.hr3_flux_x_D, self.hr3_flux_y_D,
                                 self.cell_D, self.bnd_D, self.therm_D,
                                 self.velX_D, self.velY_D).wait()
            
        # perform combination step
        self.prg.RK3_COMBINE(self.queue, self.global_size, self.local_size,
                             self.f_D, self.fr2_D, self.fr3_D,
                             self.h_D, self.hr2_D, self.hr3_D,
                             self. cell_D, self.bnd_D, self.therm_D,
                             self.velX_D, self.velY_D, self.rho_D, self.rho3_D,
                             self.ux_D, self.ux3_D, self.uy_D, self.uy3_D,
                             self.T_D, self.T3_D, self.fr2_flux_x_D,
                             self.fr3_flux_x_D, self.fr2_flux_y_D,
                             self.fr3_flux_y_D, self.hr2_flux_x_D,
                             self.hr3_flux_x_D, self.hr2_flux_y_D,
                             self.hr3_flux_y_D).wait()      

    def runSimStep(self):
        """
        perform one simulation step
        """
        
        if self.stop != 0:
            # do stuff if simulation has to be stopped
            print("simulation has stopped")
        else:
            # proceed as normal
            self.step += 1
            
            # perform requested simulation routine
            if self.RKMETHOD == 0:
                self.RK1()
            elif self.RKMETHOD == 1:
                self.RK3()
      
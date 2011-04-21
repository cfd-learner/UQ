"""
LBM

class containing all methods required for simulation of the lattice Boltzmann
method
"""

from math import *
import numpy as np

from enthought.mayavi import mlab
import time

from datetime import datetime
import os
import h5py

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
        
        platform = ['NVIDIA CUDA','ATI Stream']
        deviceName = ['GeForce GTX 480','RV710','Intel']
        
        # select device
        for found_platform in cl.get_platforms():
            if found_platform.name == platform[1]:
                my_platform = found_platform;
        
        for found_device in my_platform.get_devices():
            if found_device.name[0:5] == deviceName[2]:
            #if found_device.name == deviceName[0]:
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
        self.Pr = simSetup.Pr       #Prandtl number
        
        #viscosity model
        self.mu_model = simSetup.mu_model
        if self.mu_model == 'constant':
            self.mu = simSetup.mu    #dynamic viscosity
        elif self.mu_model == 'VHS':
            self.mu_ref = simSetup.mu_ref
            self.upsilon = simSetup.upsilon
            self.T_ref = simSetup.T_ref
        elif self.mu_model == 'sutherland':
            self.mu_ref = simSetup.mu_ref
            self.T_ref = simSetup.T_ref
            self.S_v = simSetup.S_v
        
        
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
        self.ref_rho = simSetup.ref_rho
        self.ref_T = simSetup.ref_T
        self.ref_mu = simSetup.ref_mu
        self.Tc = simSetup.Tc
        self.nPrintOut = simSetup.nPrintOut
        self.isSolid = simSetup.isSolid
        self.saveData = simSetup.saveData
        
        ##########################
        ## LISTS
        
        self.bnd = simSetup.bnd
        self.cell =  simSetup.cell
        
        self.numProps = simSetup.numProps
        self.density =   simSetup.density
        self.therm =      simSetup.therm
        self.velX = simSetup.velX
        self.velY = simSetup.velY

        ##########################
        ## REFINE INPUT
        self.setupVars()
        
        ##########################
        ## INITIALISE DIST. FUNCTIONS
        self.initFunctions()
        
        
        
        
        ##########################
        ## PLOTTING
        
        #cl.enqueue_read_buffer(self.queue, self.rho_D, self.rho_H).wait()
        #
        #mlab.surf(self.rho_H)
        #mlab.view(0,180)
        #mlab.show()
        
        # and you're good to go!
        
        ##########################
        ## SIMULATION RUN FOLDER
        
        if self.saveData:
            d = datetime.today()
            dstring = d.strftime('%Y-%m-%d_%H-%M-%S')
            currentPath = os.getcwd()
            str_list = [currentPath,'\\Results\\',dstring]
            
            self.resultsPath = ''.join(str_list)
            
            if not os.path.exists(self.resultsPath):
                os.makedirs(self.resultsPath)
   
    def setupVars(self):
        """
        setup all of the variables as required
        """
        
        # compressible gas variables
        self.D = 2.0;    #dimensionality of sim
        self.b = 2.0/(self.gamma -1.0)   #total number of D.o.F
        self.K = self.b - self.D       # number of D.o.F to solve for
        
        # reference values
        self.tau_ref = self.ref_mu/(self.R*self.ref_rho*self.ref_T)      # reference relaxation time
        self.u_ref = sqrt(self.R * self.ref_T)  #reference velocity
        self.t_ref = max(self.Lx,self.Ly)/self.u_ref    # reference time
        
        # simulation variables
        self.c = sqrt(self.R * self.Tc)     # characteristic sound speed, m/s
        
        # expand velocity lattice using chracteristic temperature
        self.ex = self.ex*self.c;
        self.ey = self.ey*self.c;
        
        # stability requirements
        
        if self.dtau == 0:
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
        elif self.tol > 0:
            self.steps = 0
        
        if need_steps == 1:
            self.steps = int(np.ceil(self.ttotal / self.dt))
            print('will need {} steps'.format(self.steps))
    
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
        self.f_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.f_H)
        self.h_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.h_H)
        
        self.dx_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.dx)
        self.dy_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.dy)
        
        self.rho_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.rho_H)
        self.ux_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.ux_H)
        self.uy_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.uy_H)
        self.T_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.T_H)
        
        self.bnd_D = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf = self.bnd)
        self.cell_D = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf = self.cell)
        
        self.density_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.density)
        self.velX_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.velX)
        self.velY_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.velY)
        self.therm_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.therm)
        
        # local
        self.fr1_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr1_H)
        self.hr1_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr1_H)
        
        if self.RKMETHOD == 0:
            self.fr1_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr1_flux_x_H)
            self.fr1_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr1_flux_y_H)
            self.hr1_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr1_flux_x_H)
            self.hr1_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr1_flux_y_H)
        
        elif self.RKMETHOD == 1:
            self.fr2_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr2_H)
            self.hr2_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr2_H)
            
            self.fr3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr3_H)
            self.hr3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr3_H)
        
            self.rho3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.rho3_H)
            self.ux3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.ux3_H)
            self.uy3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.uy3_H)
            self.T3_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.T3_H)
            
            self.fr2_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr2_flux_x_H)
            self.fr2_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr2_flux_y_H)
            self.hr2_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr2_flux_x_H)
            self.hr2_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr2_flux_y_H)
            
            self.fr3_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr3_flux_x_H)
            self.fr3_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.fr3_flux_y_H)
            self.hr3_flux_x_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr3_flux_x_H)
            self.hr3_flux_y_D = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf = self.hr3_flux_y_H)
            
        ##########################
        ## PACK DICTIONARY
        
        input = {'Nx':self.Nx,'Ny':self.Ny,'Ni':self.Ni,'R':self.R,'K':self.K,\
                 'b':self.b,'Tc':self.Tc,'dt':self.dt,\
                 'periodicX':self.periodicX,\
                 'periodicY':self.periodicY,'mirrorN':self.mirrorNorth,\
                 'mirrorS':self.mirrorSouth,'mirrorE':self.mirrorEast,\
                 'mirrorW':self.mirrorWest,'FMETHOD':self.FMETHOD,\
                 'Pr':self.Pr,'isSolid':self.isSolid,\
                 'RKMETHOD':self.RKMETHOD}
        
        if self.mu_model == 'constant':
            viscosity = {'model':self.mu_model,'mu':self.mu}
        elif self.mu_model == 'VHS':
            viscosity = {'model':self.mu_model,'T_ref':self.T_ref,\
                         'mu_ref':self.mu_ref,'upsilon':self.upsilon}
        elif self.mu_model == 'sutherland':
            viscosity = {'model':self.mu_model,'T_ref':self.T_ref,\
                         'mu_ref':self.mu_ref,'S_v':self.S_v}
        
        ##########################
        ## GENERATE ENTIRE OPENCL CODE HERE
        import codeCL
        name = codeCL.genOpenCL(input, viscosity ,self.ex, self.ey, self.dx, self.dy)
        f = open(name,'r')
        fstr = "".join(f.readlines())
        f.close()
        #print fstr
        
        t0 = time.clock()
        self.prg = cl.Program(self.ctx, fstr).build()
        t = time.clock() - t0
        
        print('total build time time = {}'.format(t))
        
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
        
        #print 'fr1'
        #cl.enqueue_read_buffer(self.queue, self.fr1_D, self.fr1_H).wait()
        #print self.fr1_H
        
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
            
        #print 'flux x'
        #cl.enqueue_read_buffer(self.queue, self.fr1_flux_x_D, self.fr1_flux_x_H).wait()
        #print self.fr1_flux_x_H
        #
        #print 'flux y'
        #cl.enqueue_read_buffer(self.queue, self.fr1_flux_y_D, self.fr1_flux_y_H).wait()
        #print self.fr1_flux_y_H
        #    
        #print 'f before update'
        #cl.enqueue_read_buffer(self.queue, self.f_D, self.f_H).wait()
        #print self.f_H
        
        # perform combination step
        self.prg.RK1_COMBINE(self.queue, self.global_size, self.local_size,
                             self.f_D, self.fr1_D, self.h_D, self.hr1_D,
                             self. cell_D, self.bnd_D, self.therm_D,
                             self.velX_D, self.velY_D, self.rho_D, self.ux_D,
                             self.uy_D, self.T_D, self.fr1_flux_x_D,
                             self.fr1_flux_y_D, self.hr1_flux_x_D,
                             self.hr1_flux_y_D).wait()
        
        #print 'f after update'
        #cl.enqueue_read_buffer(self.queue, self.f_D, self.f_H).wait()
        #print self.f_H
        
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
    
    def saveHDF5(self, all = 0):
        """
        save data to a hdf5 file
        """
        str_list = []
        str_list.append(self.resultsPath)
        str_list.append('\\step_{}'.format(self.step))
        str_list.append('.hdf5')
        
        newFile = ''.join(str_list)
        
        print newFile
        
        f = h5py.File(newFile, 'w') #open new file to save to
        
        cl.enqueue_read_buffer(self.queue, self.rho_D, self.rho_H).wait()
        f["rho"] = self.rho_H
        
        cl.enqueue_read_buffer(self.queue, self.T_D, self.T_H).wait()
        f["T"] = self.T_H
        
        cl.enqueue_read_buffer(self.queue, self.ux_D, self.ux_H).wait()
        f["ux"] = self.ux_H
        
        cl.enqueue_read_buffer(self.queue, self.uy_D, self.uy_H).wait()
        f["uy"] = self.uy_H
    
        f["t"] = self.step*self.dt
        
        f["gamma"] = self.gamma
        
        f["R"] = self.R
        
        f["T_ref"] = self.T_ref
        
        f["rho_ref"] = self.rho_ref
        
        f["dx"] = self.dx
        
        f["dy"] = self.dy
        
        f["Nx"] = self.Nx
        
        f["Ny"] = self.Ny
        
        if all == 1:
            #save all data
            cl.enqueue_read_buffer(self.queue, self.f_D, self.f_H).wait()
            f["f"] = self.f_H
            
            cl.enqueue_read_buffer(self.queue, self.h_D, self.h_H).wait()
            f["h"] = self.h_H
        
        f.close()

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
            
            if ((self.step%self.nPrintOut == 0) | (self.step == self.steps))\
            & (self.saveData == 1):
                if self.step == self.steps:
                    self.saveHDF5(1)
                else:
                    self.saveHDF5()
    
        
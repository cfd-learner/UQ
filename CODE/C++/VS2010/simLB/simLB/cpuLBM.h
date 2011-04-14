// all lattice boltzmann code for the CPU here

#ifndef CPULBM_H_
#define CPULBM_H_

#include <string>
#include <math.h>
#include <mat.h>
#include "vector_types.h"
#include "fileHandling.h"
#include "inlineFunctions.cpp"
#include "macros.h"

#define Ni 13 

class cpuLBM
{
public:

	////////////////
	// CLASS
	////////////////

	cpuLBM(folderSystem* files);
	~cpuLBM();

	////////////////
	// VARIABLES
	////////////////

	int argc_;
	char **argv_;

	int Nx;
	int Ny;
	int* bndArray;

	int RKMETHOD;						// runge-kutta method
	int FMETHOD;						// flux method
	double CFL;							// CFL number of simulation
	double dtau;
	int PROCESSOR;						// processor selection 0 = CPU, 1 = GPU, 2 = CPU + GPU
	double tt_tref;						// time of sim ratio tref
	double tt_time;						// time of sim in seconds
	int steps;							// iteration limiter
	double tol;							// tolerance
	int periodicX, periodicY;			// periodicity of domain
	int mirrorNorth, mirrorSouth;		// reflective boundary conditions
	int mirrorEast, mirrorWest;
	double gamma, R, mu, Pr;			// gas variables
	double dx,dy;
	double Lx, Ly;
	double rho_ref,p_ref,T_ref;	// reference variables
	double Tc_Tref;						// ratio Tc to Tref
	
	int VTK, MAT, TEC;
	int nPrintOut;				// print results every nPrintOut steps

	string saveToMAT;		// path to save files to
	string saveToTEC;

	string display;	// set what is to be displayed in the OpenGL visualisation

	// switches
	int pause;
	int stop;
	
	// run time parameters
	int nsteps;
	int step;			// current step number, start at 0
	double tstep;		// current time of step, s, start at 0s
	double residual;
	double initRes[100];
	double resNorm;

	//simulation variables
	double dt;
	double2 e[13];
	double b;			//constant to give correct gamma value
	double K;			// internal degrees of freedom
	double Tc;			// characteristic temperature, K


	////////////////
	// STORAGE - LOCAL
	////////////////

	// added _G as reminder that these values are effectively global

	double* f_G;
	double* h_G;

	double* rho_G;
	double2* u_G;
	double* p_G;
	double* T_G;

	// copy of original data
	double* res_copy;

	int numProps;		// number of different flow properties

	double* density;
	double* pressure;
	double* temp;
	double* velocityX;
	double* velocityY;
	int* cellType;

	////////////////
	// FUNCTIONS
	////////////////

	int loadInput(folderSystem* files);
	void initSim();
	void updateStepCount();
	int runOne(int res = 0);

	int createMAT(string optional, int DF = 0);

private:
	//compressible gas variables
	int D;				//dimensionality of simulation
	double c;			// sound speed, m/s

	//reference values
	double tau_ref;		// relaxation time, s
	double u_ref;		// reference velocity
	double t_ref;		// reference time

	

	// simulation parameters
	double ttotal;
	int total_print;

	// run time parameters
	int hasFailed;		// flag to indicate failure, 1 = equilibrium function

	int TEC_RUN;	// switch to indicate if a tec file is being written

	////////////////
	// FUNCTIONS
	////////////////

	void setupVars();
	void refineInput();
	void initialiseFunctions();

	double dot(double a1, double a2, double b1, double b2);
	int sign(double a);

	int index(int i, int pm, int d, int& mir);
	void mirIndex(int i, int mir_x, int mir_y, int& i_x, int& i_y);
	void stencil(double* f, int x, int y, int i, int lengthS, double* Sx, double* Sy);
	double minmod(double a, double b);

	void equilibrium2D(double rho, double u_, double v_, double T_, double eqf[], double eqh[]);
	void macroPropShort(int x, int y);
	void macroProp(int x, int y, double* rho, double2* u, double* T, double* p, double* tauf, double* tauh, double* tauhf);
	void macroProp3(double2* fr2_flux, double2* hr2_flux, int x, int y, double* rho, double2* u, double* T, double* p, double* tauf, double* tauh, double* tauhf);

	double NND(double ee, double* S);
	double WENO5(double ee, double* S);
	void computeFlux(double* f, int x, int y, int i, double2& flux);
	double cFLUX(double2* fluxOut, int x, int y, int i);
	void posFluxes(double* f, double* h, double2* f_flux, double2* h_flux);
	void wallFluxes(double* f, double2* f_flux, double* h, double2* h_flux);
	void boundaryFunctions(double* f, double* h, double2* f_flux, double2* h_flux);
	
	void RK1();
	void RK3();
	void EULER();
	void EULER_EQ();
};

#endif